struct ReshapeModel
    output_shape
end
Flux.@functor ReshapeModel
Flux.trainable(::ReshapeModel) = ()

(f::ReshapeModel)(input; kwargs...) = reshape(input, f.output_shape)

struct TabularModel
    params::Tuple
    TabularModel(x, y) = new(Tuple(2π .* rand(y) .- π for _ in 1:2^x))
end

Flux.@functor TabularModel

function (f::TabularModel)(input)
    input = Int.(input)
    @assert all(0 .<= input .<= 1)
    i = @Zygote.ignore 1 + parse(Int, join(input), base=2)
    return f.params[i]
end

struct LeftTabularModel
    models::Tuple
    offset::Integer
    LeftTabularModel(x, y; offset=0) = new(Tuple(TabularModel(min(i+offset,x), y) for i in 1:x), offset)
end
Base.length(f::LeftTabularModel) = length(f.models)
Flux.@functor LeftTabularModel

function (f::LeftTabularModel)(input::Vector)
    x = length(f)
    out_vecs = Tuple(f.models[i](input[1:min(i+f.offset,x)]) for i in 1:x)
    out = hcat(out_vecs...)
    return transpose(out)
end

function (f::LeftTabularModel)(input::Matrix)
    return reshape(hcat(Tuple(f(input[:,i]) for i in 1:size(input, 2))...), size(input, 1), :, size(input, 2))
end
struct BiasModel
    bias::Vector{Float64}
end
BiasModel(size::Integer) = BiasModel(2pi .* rand(size))
Flux.@functor BiasModel
Flux.trainable(m::BiasModel) = (m.bias,)

(f::BiasModel)(input; kwargs...) = f.bias

# struct SwiGlu 
struct SwiGlu
    model_in
    model_glue
    model_out
end
Flux.@functor SwiGlu

function SwiGlu(ds::Pair{<:Integer,<:Integer}, hidden_dim::Integer;
    model_in=Dense(ds[1] => hidden_dim, bias=false),
    model_glue=Dense(ds[1] => hidden_dim, bias=false),
    model_out=Dense(hidden_dim => ds[2], bias=false)
    )

    return SwiGlu(model_in, model_glue, model_out)
end

function (self::SwiGlu)(input; kwargs...)
    swi = swish(self.model_in(input))
    x_V = self.model_glue(input)
    x = x_V .* swi
    return self.model_out(x)
end


getproperty(x::SwiGlu, f::Symbol) = getfield(x.model_glue, f)

# struct ResConnection
ResConnection(block, act=x->x) = SkipConnection(block, (x, mx) -> act.((x + mx)))

function resnet_block_conv(ch_in, act; size=3, ch_middle=ch_in, kwargs...)
    c1 = Conv((size,), ch_in => ch_middle, act; pad=Flux.SamePad(), kwargs...)
    c2 = Conv((size,), ch_middle => ch_in; pad=Flux.SamePad(), kwargs...)
    return ResConnection(Chain(c1, c2), act)
end

function resnet_block_dense(f_in, act; f_mid=f_in, kwargs...)
    c1 = Dense(f_in => f_mid, act; kwargs...)
    c2 = Dense(f_mid => f_in; kwargs...)
    return ResConnection(Chain(c1, c2), act)
end

include("recurrent.jl")