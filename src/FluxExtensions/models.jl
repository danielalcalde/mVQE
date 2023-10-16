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