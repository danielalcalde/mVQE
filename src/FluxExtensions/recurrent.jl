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
    x1 = permutedims(x1, (3, 1, 2))
    
    x1 = self.conv(x1) # shape (L, d_out, batch)
    x1 = swish_.(x1)
    
    x1 = permutedims(x1, (2, 3, 1)) #shape (d_out, batch, L)
    
    Flux.reset!(self.rnn)
    y = self.rnn(x1) #shape (d_out, batch, L)

    if self.rnn_reverse !== nothing
        Flux.reset!(self.rnn_reverse)
        y2 = self.rnn_reverse(x1[:, :, end:-1:1])  
        y = vcat(y, y2) # shape (2*d_out, batch, L)
    end
    
    output = self.dense_out(y) .+ x2 #shape (d_out, batch, L)
    return output
end

function RNNCNNBlock(s::Pair{<:Integer,<:Integer}; d_conv=3, RNN_type=Flux.RNN, bidirectional=false, rms_norm=true)
    rnn2 = nothing
    dense_dim = s.second
    if bidirectional
        rnn2 = RNN_type(s.second => s.second)
        dense_dim = 2 * s.second
    end

    rms_norm_ = nothing
    if rms_norm
        rms_norm_ = RMSNorm(s.second)
    end
    return RNNCNNBlock(
            Dense(s.first => s.second * 2),
            Conv((d_conv,), s.second => s.second; pad=Flux.SamePad()),
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