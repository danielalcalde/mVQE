module Circuits

using PastaQ
using ITensors
using Random
using OptimKit
using Zygote
using Statistics

using ITensors: AbstractMPS

using ..ITensorsExtension: VectorAbstractMPS, States, projective_measurement
using ..Layers: Rylayer, CXlayer, Rxlayer, ProjectiveMeasurementLayer

# Types
circuit_type = Vector{Tuple}

abstract type AbstractCircuit end
abstract type AbstractVariationalCircuit end

struct VariationalCircuitRy <: AbstractVariationalCircuit 
    N:: Int
    depth:: Int
end
initialize_circuit(model::VariationalCircuitRy) = [2π .* rand(model.N) for _ in 1:model.depth]

Base.show(io::IO, c::VariationalCircuitRy) = print(io, "VariationalCircuitRy(N=$(c.N), depth=$(c.depth))")

function generate_circuit(model::VariationalCircuitRy, θ)
    circuit = Tuple[]
    @assert length(θ) == model.depth

    for (d, θi) in enumerate(θ)
      circuit = vcat(circuit, CXlayer(model.N, d))
      circuit = vcat(circuit, Rylayer(θi))
    end
    return Circuit(circuit)
end


struct VariationalMeasurement <: AbstractVariationalCircuit
    vcircuit:: AbstractVariationalCircuit
    depth:: Int
    measurement_indices:: Vector{Int}
    reset:: Int
end

initialize_circuit(model::VariationalMeasurement) = [initialize_circuit(model.vcircuit) for _ in 1:model.depth]

function generate_circuit(model::VariationalMeasurement, θs)
    circuit = Tuple[]
    @assert length(θs) == model.depth
    for θ in θs
        circuit = vcat(circuit, generate_circuit(model.vcircuit, θ).circ)
        circuit = vcat(circuit, ProjectiveMeasurementLayer(model.measurement_indices, model.reset))
    end

    return Circuit(circuit)
end

# Old
struct OldMeasurementVariationalCircuitRy <: AbstractVariationalCircuit 
    N:: Int
    depth:: Int
    k:: Int
    measurement_indices:: Vector{Int}
end
initialize_circuit(model::OldMeasurementVariationalCircuitRy) = [[2π .* rand(model.N) for _ in 1:model.depth] for _ in 1:model.k]

function generate_circuit(model::OldMeasurementVariationalCircuitRy, θs)
    circuits = Tuple{circuit_type, Vector{Int}}[]
    @assert length(θs) == model.k

    for θ in θs
        circuit = Tuple[]
        for d in 1:model.depth
            circuit = vcat(circuit, CXlayer(model.N, d))
            circuit = vcat(circuit, Rylayer(θ[d]))
        end
        circuits = vcat(circuits, [(circuit, model.measurement_indices)])
    end
    return MeasurementCircuit(circuits)
end

# Circuit structers
struct Circuit <: AbstractCircuit
    circ :: Vector{Tuple}
end

struct MeasurementCircuit <: AbstractCircuit
    circ :: Vector{Tuple{circuit_type, Vector{Int}}}
end

# Evolve wave function with a sequence of gates
function PastaQ.runcircuit(ψ::States, model::AbstractVariationalCircuit, θ; kwargs...)
    circuit = generate_circuit(model, θ)
    return runcircuit(ψ, circuit; kwargs...)
end

PastaQ.runcircuit(ψs::VectorAbstractMPS, circuit::AbstractCircuit; kwargs...) = 
    [runcircuit(ψ, circuit; kwargs...) for ψ in ψs]

function PastaQ.runcircuit(ψ::AbstractMPS, circuit::Circuit; kwargs...)
    #println(circuit.circ)
    return runcircuit(ψ, circuit.circ; kwargs...)
end


#Currently the backpropagation does not work properly since there is an error in the  function ChainRulesCore.rrule(::typeof(apply)
function PastaQ.runcircuit(ρ::AbstractMPS, circuits::MeasurementCircuit; kwargs...)
    for circuit in circuits.circ
        circuit, measurement_indices = circuit
        ρ = runcircuit(ρ, circuit; kwargs...)
        ρ, = projective_measurement(ρ; indices=measurement_indices, reset=1)
    end
    return ρ
end



#end
end