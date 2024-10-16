function rms_norm(x::AbstractArray; dims=ndims(x), ϵ=Flux.ofeltype(x, 1e-5))
    σ = mean(x .^ 2, dims=dims)
    return x ./ sqrt.(σ .+ ϵ)
end

struct RMSNorm
    weight::AbstractArray
end
function RMSNorm(i::Integer)
    return RMSNorm(ones(Float32, i))
end

Flux.@functor RMSNorm
function (f::RMSNorm)(input; kwargs...)
    input = rms_norm(input; dims=(1)) .* f.weight
    return input
end

struct ReshapeAndPermuteRNN
    model
end
Flux.@functor ReshapeAndPermuteRNN
Flux.trainable(a::ReshapeAndPermuteRNN) = (a.model,)
function ReshapeAndPermuteRNN(models...)
    if models[1] isa Chain
        models = models[1]
    else
        models = Chain(models...)
    end
    return ReshapeAndPermuteRNN(models)
end

function (f::ReshapeAndPermuteRNN)(input; kwargs...)
    dim = length(size(input))
    if dim == 1
        input = reshape(input, 1, 1, size(input, 1)) # input_dim (1, 1, L)
    elseif dim == 2
        input = reshape(input, 1, size(input, 1), size(input, 2)) # input_dim (1, L, batch)
        input = permutedims(input, (1, 3, 2)) # input_dim (1, batch, L)
    end
        
    input = input[:, :, end:-1:1]
    Flux.reset!(f.model)
    output = f.model(input)
    output = output[:, :, end:-1:1]
    output = permutedims(output, (3, 1, 2))
    
    if dim == 1
        output = output[:, :, 1] # output_dim (L, dim)
    #else # output_dim (L, dim, batch)
    end

    return output
end


struct RNNCNNBlock
    dense_in
    conv
    rnn
    rnn_reverse
    dense_out
    rms_norm
end
Flux.@functor RNNCNNBlock

function (self::RNNCNNBlock)(input; kwargs...)
    swish_ = NNlib.fast_act(swish, input)
    
    x = input #shape (d_in, batch, L)
    if self.rms_norm !== nothing
        x = self.rms_norm(x)
    end
    
    x = self.dense_in(x) #shape (2*d_out, batch, L)
    d_out = size(x, 1) ÷ 2
    x1, x2 = x[1:d_out, :, :], x[d_out + 1:end, :, :]
    
    if self.conv !== nothing
        x1 = permutedims(x1, (3, 1, 2))
        x1 = self.conv(x1) # shape (L, d_out, batch)
        x1 = swish_.(x1)
        x1 = permutedims(x1, (2, 3, 1)) #shape (d_out, batch, L)
    end
    
    Flux.reset!(self.rnn)
    y = self.rnn(x1) #shape (d_out, batch, L)

    if self.rnn_reverse !== nothing
        Flux.reset!(self.rnn_reverse)
        y2 = self.rnn_reverse(x1[:, :, end:-1:1])
        y2 = y2[:, :, end:-1:1]
        y = vcat(y, y2) # shape (2*d_out, batch, L)
    end
    
    output = self.dense_out(y) .+ x2 #shape (d_out, batch, L)
    return output
end

function RNNCNNBlock(s::Pair{<:Integer,<:Integer}; d_conv=3, RNN_type=Flux.RNN, no_conv=false, bidirectional=false, rms_norm=true)
    rnn2 = nothing
    dense_dim = s.second
    if bidirectional
        rnn2 = RNN_type(s.second => s.second)
        dense_dim = 2 * s.second
    end
    if no_conv
        conv = nothing
    else
        conv = Conv((d_conv,), s.second => s.second; pad=Flux.SamePad())
    end

    rms_norm_ = nothing
    if rms_norm
        rms_norm_ = RMSNorm(s.second)
    end
    return RNNCNNBlock(
            Dense(s.first => s.second * 2),
            conv,
            RNN_type(s.second => s.second),
            rnn2,
            Dense(dense_dim => s.second),
            rms_norm_
            )
end

function Base.show(io::IO, l::RNNCNNBlock)
    dconv = size(l.conv.weight, 1)
    print(io, "RNNCNNBlock(", size(l.dense_in.weight, 1), " => ", size(l.dense_out.weight, 2), ", dconv=", dconv ,")")
end

struct RNNCNNLightBlock
    conv
    rnn
    rnn_reverse
    dense_out
    rms_norm
end
Flux.@functor RNNCNNLightBlock

function (self::RNNCNNLightBlock)(input; kwargs...)
    # Conv Block
    x1 = input #shape (d_in, batch, L)
    if self.conv !== nothing
        if self.rms_norm !== nothing
            x1 = self.rms_norm(x1)
        end
        input += self.conv(x1)
    end

    # Rnn Block
    x2 = input #shape (d_in, batch, L)
    if self.rms_norm !== nothing
        x2 = self.rms_norm(x2)
    end

    Flux.reset!(self.rnn)
    x2 = self.rnn(x2) #shape (d_out, batch, L)

    if self.rnn_reverse !== nothing
        Flux.reset!(self.rnn_reverse)
        x2_2 = self.rnn_reverse(input[:, :, end:-1:1])  
        x2_2 = x2_2[:, :, end:-1:1]
        x2 = vcat(x2, x2_2) # shape (2*d_out, batch, L)
        x2 = self.dense_out(x2) #shape (d_out, batch, L)
    end
    input += x2
    return input
end

function RNNCNNLightBlock(s::Pair{<:Integer,<:Integer}; d_conv=3, no_conv=false, hidden_conv_dim = s.second, RNN_type=Flux.RNN, bidirectional=false, rms_norm=true)
    @assert s.first == s.second
    rms_norm_ = nothing
    if rms_norm
        rms_norm_ = RMSNorm(s.second)
    end

    # Conv
    if no_conv
        conv_swiglu = nothing
    else
        conv = ConvRNNOrder((d_conv,), s.second => hidden_conv_dim; pad=Flux.SamePad())
        conv_swiglu = SwiGlu(s.first => s.second, hidden_conv_dim; model_in=conv)
    end

    # RNN
    rnn = RNN_type(s.second => s.second)
    rnn2 = nothing
    dense_out = nothing
    if bidirectional
        rnn2 = RNN_type(s.second => s.second)
        dense_out = Dense(2 * s.second => s.second)
    end

    return RNNCNNLightBlock(
            conv_swiglu,
            rnn,
            rnn2,
            dense_out,
            rms_norm_
            )
end

function Base.show(io::IO, l::RNNCNNLightBlock)
    dconv = size(l.conv.weight, 1)
    print(io, "RNNCNNLightBlock(", size(l.dense_in.weight, 1), " => ", size(l.dense_out.weight, 2), ", dconv=", dconv ,")")
end


### RNNLLamaBlock
struct RNNSwiGluBlock
    swiglu
    rnn
    rnn_reverse
    dense_out
    rms_norm
end
Flux.@functor RNNSwiGluBlock

function (self::RNNSwiGluBlock)(input; kwargs...)
    
    # Rnn Block
    x = input #shape (d_in, batch, L)
    if self.rms_norm !== nothing
        x = self.rms_norm(x)
    end

    Flux.reset!(self.rnn)
    x_1 = self.rnn(x) #shape (d_out, batch, L)

    if self.rnn_reverse !== nothing
        Flux.reset!(self.rnn_reverse)
        x_2 = self.rnn_reverse(x[:, :, end:-1:1])  
        x_2 = x_2[:, :, end:-1:1]
        x = vcat(x_1, x_2) # shape (2*d_out, batch, L)
        x = self.dense_out(x) #shape (d_out, batch, L)
    end
    input += x

    # Swiglu Block
    x = input #shape (d_in, batch, L)
    if self.swiglu !== nothing
        if self.rms_norm !== nothing
            x = self.rms_norm(x)
        end
        input += self.swiglu(x)
    end

    return input
end

function RNNSwiGluBlock(s::Pair{<:Integer,<:Integer}; swiglu=true, hidden_dim=round(Int, 2.5 * s.second), RNN_type=Flux.RNN, bidirectional=false, double_dense=true, rms_norm=true)
    @assert s.first == s.second
    rms_norm_ = nothing
    if rms_norm
        rms_norm_ = RMSNorm(s.second)
    end

    # RNN
    rnn = RNN_type(s.first => s.second)
    rnn2 = nothing
    dense_out = nothing
    if bidirectional
        rnn2 = RNN_type(s.first => s.second)
        dense_out = Dense(2 * s.second => s.second)
    end

    # SwiGlu
    if !swiglu
        swiglu = nothing
    else
        swiglu = SwiGlu(s.second => s.second, hidden_dim)
    end

    return RNNSwiGluBlock(
            swiglu,
            rnn,
            rnn2,
            dense_out,
            rms_norm_
            )
end

function Base.show(io::IO, l::RNNSwiGluBlock)
    hidden_dim = size(l.swiglu.model_in.weight, 1)
    print(io, "RNNSwiGluBlock(", size(l.swiglu.model_in.weight, 2), " => ", size(l.swiglu.model_out.weight, 1), ", hidden_dim=", hidden_dim ,")")
end



struct RNNOrder
    conv
    global ConvRNNOrder
    RNNOrder(o) = new(o)
    ConvRNNOrder(args...; kwargs...) = new(Conv(args...; kwargs...))
end
Flux.@functor RNNOrder

function (self::RNNOrder)(input; kwargs...)
    x = input # shape (d_in, batch, L)
    x = permutedims(x, (3, 1, 2))
    x = self.conv(x) # (L, d_out, batch)
    x = permutedims(x, (2, 3, 1))
    return x # shape (d_out, batch, L)
end