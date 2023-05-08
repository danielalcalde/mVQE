module Circuits

using PastaQ
using ITensors
using Random
using OptimKit
using Zygote
using Statistics
using Flux
import Base

using ITensors: AbstractMPS

using ..ITensorsExtension: VectorAbstractMPS, States, projective_measurement, projective_measurement_sample
using ..Layers: Rxlayer, Rylayer, Rzlayer, CXlayer, CRxlayer
using ..Layers: ProjectiveMeasurementLayer

# Types
CircuitType = Vector{Tuple}

abstract type AbstractVariationalCircuit end

function Base.size(model::AbstractVariationalCircuit)
    throw("Size not defined for $(typeof(model)). This should be the size of the parameters.")
end
function Base.size(model::AbstractVariationalCircuit, i::Int)
    throw("Size not defined for $(typeof(model))")
end

# Variational circuit with Ry gates
struct VariationalCircuitRy <: AbstractVariationalCircuit
    params::Matrix{Float64}
    VariationalCircuitRy(params::Matrix{Float64}) = new(params)
    VariationalCircuitRy(N::Int, depth::Int) = new(2π .* rand(N, depth))
    VariationalCircuitRy() = new(Matrix{Float64}(undef, 0, 0)) # Empty circuit to be used as a placeholder
end
Flux.@functor VariationalCircuitRy
Base.size(model::VariationalCircuitRy) = size(model.params)
Base.size(model::VariationalCircuitRy, i::Int) = size(model.params, i)
Base.show(io::IO, c::VariationalCircuitRy) = print(io, "VariationalCircuitRy(N=$(size(c, 1)), depth=$(size(c, 2)))")
get_depth(model::VariationalCircuitRy) = size(model.params, 2)
get_N(model::VariationalCircuitRy) = size(model.params, 1)

function generate_circuit(model::VariationalCircuitRy; params=nothing)
    if params === nothing
        @assert size(model.params, 1) > 0 "VariationalCircuitRy is empty"
        params = model.params
    end

    circuit = Tuple[]

    N = N = get_N(model)
    depth = get_depth(model)

    for d in 1:depth
        circuit = vcat(circuit, CXlayer(N, d))
        circuit = vcat(circuit, Rylayer(params[:, d]))
    end
    return circuit
end

# Variational circuit with Ry, Rx, Rz and CRx gates
struct VariationalCircuitOverparametrizied <: AbstractVariationalCircuit
    params::Array{Float64, 3}
    VariationalCircuitOverparametrizied(params::Array{Float64, 3}) = new(params)
    VariationalCircuitOverparametrizied(N::Int, depth::Int) = new(2π .* rand(N, 4, depth))
    VariationalCircuitOverparametrizied() = new(Array{Float64, 3}(undef, 0, 0, 0)) # Empty circuit to be used as a placeholder
end
Flux.@functor VariationalCircuitOverparametrizied 
Base.size(model::VariationalCircuitOverparametrizied) = size(model.params)
Base.size(model::VariationalCircuitOverparametrizied, i::Int) = size(model.params, i)
get_depth(model::VariationalCircuitOverparametrizied) = size(model.params, 3)
get_N(model::VariationalCircuitOverparametrizied) = size(model.params, 1)

function generate_circuit(model::VariationalCircuitOverparametrizied; params=nothing)
    if params === nothing
        @assert size(model.params, 1) > 0 "VariationalCircuitOverparametrizied is empty"
        params = model.params
    end

    circuit = Tuple[]

    N = get_N(model)
    depth = get_depth(model)

    for d in 1:depth
        circuit = vcat(circuit, Rxlayer(params[:, 1, d]))
        circuit = vcat(circuit, Rylayer(params[:, 2, d]))
        circuit = vcat(circuit, Rzlayer(params[:, 3, d]))
        circuit = vcat(circuit, CRxlayer(N, d, params[:, 4, d]))
    end
    return circuit
end


# Measurment circuit
abstract type AbstractVariationalMeasurementCircuit <: AbstractVariationalCircuit end
Flux.trainable(a::AbstractVariationalMeasurementCircuit) = (a.vcircuits,)
Base.length(a::AbstractVariationalMeasurementCircuit) = length(a.vcircuits)
Base.getindex(a::AbstractVariationalMeasurementCircuit, i::Int64) = a.vcircuits[i]

struct VariationalMeasurement <: AbstractVariationalMeasurementCircuit
    vcircuits:: Vector{AbstractVariationalCircuit}
    measurement_indices:: Vector{Int}
    reset:: Int
end
Flux.@functor VariationalMeasurement

VariationalMeasurement(vcircuits::Vector{T}, measurement_indices:: Vector{Int}; reset:: Int=1) where T <: AbstractVariationalCircuit = VariationalMeasurement(vcircuits, measurement_indices, reset)

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
    measurement_indices:: Vector{Int}
    reset:: Int
end
Flux.@functor VariationalMeasurementMC

VariationalMeasurementMC(vcircuits::Vector{T}, measurement_indices:: Vector{Int}; reset:: Int=1) where T <: AbstractVariationalCircuit = VariationalMeasurementMC(vcircuits, measurement_indices, reset)

function (model::VariationalMeasurementMC)(ρ::States; kwargs...)
    for vcircuit in model.vcircuits
        ρ = vcircuit(ρ; kwargs...)
        ρ, _ = projective_measurement_sample(ρ; indices=model.measurement_indices, reset=model.reset)
    end

    return ρ
end

# Run circuit

# Evolve wave function with a sequence of gates
function (model::AbstractVariationalCircuit)(ψ::States; params=nothing, kwargs...)
    circuit = generate_circuit(model; params=params)
    return runcircuit(ψ, circuit; kwargs...)
end

function (circuit::CircuitType)(ψ::AbstractMPS; kwargs...)
    return runcircuit(ψ, circuit; kwargs...)
end

function (circuit::CircuitType)(ψs::VectorAbstractMPS; kwargs...) 
    return [circuit(ψ, circuit; kwargs...) for ψ in ψs]
end


include("FeedbackCircuits/FeedbackCircuits.jl")

#end
end # module