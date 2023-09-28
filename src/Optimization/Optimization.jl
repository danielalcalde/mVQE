using ..MPOExtensions: PartialMPO


PartialMPOs = Union{PartialMPO, Vector{PartialMPO}}

MPOType = Union{AbstractMPS, PartialMPO}
MPOVectorType = Vector{T} where T <: MPOType
MPOTypes = Union{MPOType, MPOVectorType, Vector{AbstractMPS}}

include("expect.jl")
include("nMPS.jl")
include("loss_and_grad.jl")
include("optimize_and_evolve.jl")