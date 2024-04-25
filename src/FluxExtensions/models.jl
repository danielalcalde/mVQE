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

struct BiasModel
    bias::Vector{Float64}
end
BiasModel(size::Integer) = BiasModel(2pi .* rand(size))
Flux.@functor BiasModel
Flux.trainable(m::BiasModel) = (m.bias,)

(f::BiasModel)(input; kwargs...) = f.bias


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