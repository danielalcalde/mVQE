struct ReshapeModel
    output_shape
end
Flux.@functor ReshapeModel
Flux.trainable(::ReshapeModel) = ()

(f::ReshapeModel)(input; kwargs...) = reshape(input, f.output_shape)

struct BiasModel
    bias::Vector{Float64}
end
BiasModel(size::Integer) = BiasModel(2pi .* rand(size))
Flux.@functor BiasModel
Flux.trainable(m::BiasModel) = (m.bias,)

(f::BiasModel)(input; kwargs...) = f.bias