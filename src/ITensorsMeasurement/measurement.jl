module ITensorsMeasurement
using Zygote
using ITensors
using ITensors: AbstractMPS

# Types
VectorAbstractMPS = Union{Vector{MPS}, Vector{MPO}}
States = Union{VectorAbstractMPS, AbstractMPS} 

include("projective_measurement.jl")
include("projective_measurement_sample_pure.jl")
include("projective_measurement_sample_mixed.jl")



end # module