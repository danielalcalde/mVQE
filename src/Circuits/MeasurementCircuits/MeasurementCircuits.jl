abstract type AbstractVariationalMeasurementCircuit <: AbstractVariationalCircuit end
Flux.trainable(a::AbstractVariationalMeasurementCircuit) = (a.vcircuits,)
Base.length(a::AbstractVariationalMeasurementCircuit) = length(a.vcircuits)
Base.getindex(a::AbstractVariationalMeasurementCircuit, i::Int64) = a.vcircuits[i]
get_depth(model::AbstractVariationalMeasurementCircuit) = length(model.vcircuits)
get_N(model::AbstractVariationalMeasurementCircuit) = get_N(model.vcircuits[1])

struct VariationalMeasurement <: AbstractVariationalMeasurementCircuit
    vcircuits:: Vector{AbstractVariationalCircuit}
    measurement_indices:: Vector{<:Integer}
    reset:: Int
end
Flux.@functor VariationalMeasurement

VariationalMeasurement(vcircuits::Vector{T}, measurement_indices:: Vector{<:Integer}; reset:: Int=1) where T <: AbstractVariationalCircuit = VariationalMeasurement(vcircuits, measurement_indices, reset)

function generate_circuit(model::VariationalMeasurement; params=nothing)
    @assert params === nothing "VariationalMeasurement does not take parameters"
    circuit = Tuple[]
    for vcircuit in model.vcircuits
        circuit = vcat(circuit, generate_circuit(vcircuit))
        circuit = vcat(circuit, ProjectiveMeasurementLayer(model.measurement_indices, model.reset))
    end

    return circuit
end

struct VariationalMeasurementMC <: AbstractVariationalMeasurementCircuit
    vcircuits:: Vector{AbstractVariationalCircuit}
    measurement_indices:: Vector{<:Integer}
    reset:: Int
end
Flux.@functor VariationalMeasurementMC

VariationalMeasurementMC(vcircuits::Vector{T}, measurement_indices:: Vector{<:Integer}; reset:: Int=1) where T <: AbstractVariationalCircuit = VariationalMeasurementMC(vcircuits, measurement_indices, reset)

function (model::VariationalMeasurementMC)(ρ::States; kwargs...)
    for vcircuit in model.vcircuits
        ρ = vcircuit(ρ; kwargs...)
        ρ, _ = projective_measurement_sample(ρ; indices=model.measurement_indices, reset=model.reset)
    end

    return ρ
end

include("ReshapeModel.jl")
include("FeedbackCircuits.jl")