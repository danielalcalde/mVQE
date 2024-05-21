module FluxExtensions
using Flux
using Zygote
using Statistics


include("models.jl")
include("destructure.jl")
include("variance.jl")
include("tokenizer.jl")

function ExpDecayGen(lr_start::Real, lr_end::Real, steps::Integer; decay_step=1, clip=lr_end)
    eff_steps = steps / decay_step
    decay = (lr_end/lr_start) ^ (1/eff_steps)
    return ExpDecay(lr_start, decay, decay_step, clip)
end


end # module

