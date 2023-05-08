module FluxExtensions
using Flux

struct ReshapeModel
    output_shape
end
Flux.@functor ReshapeModel
Flux.trainable(::ReshapeModel) = ()

(f::ReshapeModel)(input; kwargs...) = reshape(input, f.output_shape)

end # module