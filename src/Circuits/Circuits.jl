module Circuits

using PastaQ
using ITensors
using Random
using OptimKit
using Zygote
using Statistics
using Flux

import Base
import ITensorsExtensions

using ITensors: AbstractMPS

using ..ITensorsExtension: VectorAbstractMPS, States
using ..ITensorsMeasurement: projective_measurement, projective_measurement_sample
using ..Layers: Rxlayer, Rylayer, Rzlayer, CXlayer, CRxlayer, BrickLayer, OneGateLayer, Ulayer, CUlayer, CUlayer_broken, CX_Idlayer, FullTwoBody
using ..Layers: ProjectiveMeasurementLayer

# Types
CircuitType = Vector{Tuple}
abstract type AbstractCircuit end


abstract type AbstractVariationalCircuit <: AbstractCircuit end

function Base.size(model::AbstractVariationalCircuit)
    throw("Size not defined for $(typeof(model)). This should be the size of the parameters.")
end
function Base.size(model::AbstractVariationalCircuit, i::Int)
    throw("Size not defined for $(typeof(model))")
end

"""
    vector_size(model::AbstractVariationalCircuit)
    This should be the size of the parameters as a Vector. If there are nested parameters in tuples this should follow that structure.
"""
function vector_size(model::AbstractVariationalCircuit)
    return collect(size(model))
end

function get_depth(model::AbstractVariationalCircuit)
    throw("get_depth not defined for $(typeof(model))")
end

function get_N(model::AbstractVariationalCircuit)
    throw("get_N not defined for $(typeof(model))")
end

Flux.trainable(model::AbstractVariationalCircuit) = (model.params,)
number_of_parameters(model::AbstractVariationalCircuit) = sum([length(p[:]) for p in Flux.params(model)]; init=0)
Base.show(io::IO, c::AbstractVariationalCircuit) = print(io, "$(typeof(c))(N=$(get_N(c)), depth=$(get_depth(c)); Nₚ=$(number_of_parameters(c)))")

function get_parameters(model::AbstractVariationalCircuit; params=nothing)
    if params === nothing
        @assert size(model.params, 1) > 0 "$(typeof(model)) is empty"
        params = model.params
    end

    N = get_N(model)
    depth = get_depth(model)
    return params, N, depth
end


function generate_circuit(model::AbstractVariationalCircuit; params=nothing)
    params, N, depth = get_parameters(model; params)

    circuit = Tuple[]
    return generate_circuit!(circuit, model; params, N, depth)
end

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


# Circuit Example
struct IdentityCircuit <: AbstractVariationalCircuit
    N::Int
end

get_N(model::IdentityCircuit) = model.N
get_depth(model::IdentityCircuit) = 0

(model::IdentityCircuit)(ρ::States; kwargs...) = ρ
Flux.trainable(::IdentityCircuit) = ()

function select_params(params, model_params)
    if params === nothing
        return model_params
    end
    @assert length(params) == length(model_params) "Number of parameters must match number of qubits in the layer"
    return params
end

mutable struct NonVariaitonalCircuit <: AbstractVariationalCircuit
    circuit::CircuitType
end
get_N(model::NonVariaitonalCircuit) = 0
get_depth(model::NonVariaitonalCircuit) = length(model.circuit)
get_parameters(model::NonVariaitonalCircuit; params=nothing) = nothing, 0, length(model.circuit)
Flux.trainable(::NonVariaitonalCircuit) = ()
function generate_circuit!(circuit, model::NonVariaitonalCircuit; kwargs...)
    return vcat(circuit, model.circuit)
end


include("VariationalCircuitComposed.jl")
include("VariationalCircuitR.jl")
include("VariationalCircuitSymRy.jl")
include("MeasurementCircuits/MeasurementCircuits.jl")
include("CompleteUnitary.jl")
include("VariationalCircuitTwoQubitGate.jl")
#end
end # module