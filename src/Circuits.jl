module Circuits

using PastaQ
using ITensors
using Random
using OptimKit
using Zygote
using Statistics

using ITensors: AbstractMPS

using ..ITensorsExtension: VectorAbstractMPS, States, projective_measurement, projective_measurement_sample
using ..Layers: Rylayer, CXlayer, Rxlayer, ProjectiveMeasurementLayer

# Types
CircuitType = Vector{Tuple}

abstract type AbstractVariationalCircuit end

struct VariationalCircuitRy <: AbstractVariationalCircuit 
    N:: Int
    depth:: Int
end
initialize_circuit(model::VariationalCircuitRy) = 2π .* rand(model.depth, model.N)

Base.show(io::IO, c::VariationalCircuitRy) = print(io, "VariationalCircuitRy(N=$(c.N), depth=$(c.depth))")

function generate_circuit(model::VariationalCircuitRy, θ)
    circuit = Tuple[]
    @assert size(θ, 1) == model.depth

    for d in 1:size(θ, 1)
        circuit = vcat(circuit, CXlayer(model.N, d))
        circuit = vcat(circuit, Rylayer(θ[d, :]))
    end
    return circuit
end

abstract type AbstractVariationalMeasurementCircuit <: AbstractVariationalCircuit end
struct VariationalMeasurement <: AbstractVariationalMeasurementCircuit
    vcircuit:: AbstractVariationalCircuit
    depth:: Int
    measurement_indices:: Vector{Int}
    reset:: Int
end

initialize_circuit(model::AbstractVariationalMeasurementCircuit) = [initialize_circuit(model.vcircuit) for _ in 1:model.depth]

function generate_circuit(model::VariationalMeasurement, θs)
    circuit = Tuple[]
    @assert length(θs) == model.depth
    for θ in θs
        circuit = vcat(circuit, generate_circuit(model.vcircuit, θ))
        circuit = vcat(circuit, ProjectiveMeasurementLayer(model.measurement_indices, model.reset))
    end

    return circuit
end

struct VariationalMeasurementMC <: AbstractVariationalMeasurementCircuit
    vcircuit:: AbstractVariationalCircuit
    depth:: Int
    measurement_indices:: Vector{Int}
    reset:: Int
end

function PastaQ.runcircuit(ρ::States, model::VariationalMeasurementMC, θs; kwargs...)
    @assert length(θs) == model.depth
    for i in 1:model.depth
        ρ = runcircuit(ρ, model.vcircuit, θs[i]; kwargs...)
        ρ, _ = projective_measurement_sample(ρ; indices=model.measurement_indices, reset=model.reset)
    end

    return ρ
end

# Run circuit

# Evolve wave function with a sequence of gates
function PastaQ.runcircuit(ψ::States, model::AbstractVariationalCircuit, θ; kwargs...)
    circuit = generate_circuit(model, θ)
    return runcircuit(ψ, circuit; kwargs...)
end

PastaQ.runcircuit(ψs::VectorAbstractMPS, circuit::CircuitType; kwargs...) = 
    [runcircuit(ψ, circuit; kwargs...) for ψ in ψs]


include("FeedbackCircuits/FeedbackCircuits.jl")

#end
end # module