module FluxExtensions
using Flux

struct ReshapeModel
    output_shape
end
Flux.@functor ReshapeModel
Flux.trainable(::ReshapeModel) = ()

(f::ReshapeModel)(input; kwargs...) = reshape(input, f.output_shape)


function ExpDecayGen(lr_start::Real, lr_end::Real, steps::Integer; decay_step=1, clip=lr_end)
    eff_steps = steps / decay_step
    decay = (lr_end/lr_start) ^ (1/eff_steps)
    return ExpDecay(lr_start, decay, decay_step, clip)
end



end # module

