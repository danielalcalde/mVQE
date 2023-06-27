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

using ..ITensorsExtension: VectorAbstractMPS, States
using ..ITensorsMeasurement: projective_measurement, projective_measurement_sample
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

function get_depth(model::AbstractVariationalCircuit)
    throw("get_depth not defined for $(typeof(model))")
end

function get_N(model::AbstractVariationalCircuit)
    throw("get_N not defined for $(typeof(model))")
end

Flux.trainable(model::AbstractVariationalCircuit) = (model.params,)

Base.show(io::IO, c::AbstractVariationalCircuit) = print(io, "$(typeof(c))(N=$(get_N(c)), depth=$(get_depth(c)))")

function generate_circuit(model::AbstractVariationalCircuit; params=nothing)
    if params === nothing
        @assert size(model.params, 1) > 0 "$(typeof(model)) is empty"
        params = model.params
    end

    circuit = Tuple[]

    N = N = get_N(model)
    depth = get_depth(model)

    return generate_circuit!(circuit, model; params, N, depth)
end


struct VariationalCircuitComposed <: AbstractVariationalCircuit
    circuits::Vector{AbstractVariationalCircuit}
    function VariationalCircuitComposed(circuits::Vector{T}) where T <: AbstractVariationalCircuit
        for circuit in circuits
            @assert get_N(circuit) == get_N(circuits[1]) "All circuits must have the same number of qubits"
        end
        new(circuits)
    end
end
Flux.@functor VariationalCircuitComposed
Base.size(model::VariationalCircuitComposed) = collect(size(model.circuits[i]) for i in 1:length(model.circuits))
Base.size(model::VariationalCircuitComposed, i::Int) = size(model.circuits[i])
get_N(model::VariationalCircuitComposed) = get_N(model.circuits[1])
get_depth(model::VariationalCircuitComposed) = sum(get_depth(circ) for circ in model.circuits)

function generate_circuit(model::VariationalCircuitComposed; params=nothing)
    if params === nothing
        params = [nothing for _ in model.circuits]
    else
        @assert length(params) == length(model.circuits) "Number of parameters must match number of circuits"
    end

    circuit = Tuple[]

    for (i, circ) in enumerate(model.circuits)
        circuit = vcat(circuit, generate_circuit(circ; params=params[i]))
    end
    return circuit
end

struct VariationalCircuitRx{T <: Number} <: AbstractVariationalCircuit
    params::Matrix{T}
    order::Bool
    VariationalCircuitRx(params::Matrix{T}, order=false) where T <: Number = new{T}(params, order)
    VariationalCircuitRx(N::Int, depth::Int; order=false, eltype=Float64) = new{eltype}(2π .* rand(N, depth), order)
    VariationalCircuitRx() = new{Float64}(Matrix{Float64}(undef, 0, 0), false) # Empty circuit to be used as a placeholder
end
Flux.@functor VariationalCircuitRx
Base.size(model::VariationalCircuitRx) = size(model.params)
Base.size(model::VariationalCircuitRx, i::Int) = size(model.params, i)
get_depth(model::VariationalCircuitRx) = size(model.params, 2)
get_N(model::VariationalCircuitRx) = size(model.params, 1)


function generate_circuit!(circuit, model::VariationalCircuitRx; params=nothing, N::Integer, depth::Integer)
    if model.order
        for d in 1:depth-1
            circuit = vcat(circuit, Rxlayer(params[:, d]))
            circuit = vcat(circuit, CXlayer(N, d+1))
        end

        circuit = vcat(circuit, Rxlayer(params[:, depth]))
    else
        for d in 1:depth
            circuit = vcat(circuit, CXlayer(N, d))
            circuit = vcat(circuit, Rxlayer(params[:, d]))
        end
    end

    
    return circuit
end

# Variational circuit with Ry gates
struct VariationalCircuitRy{T <: Number} <: AbstractVariationalCircuit
    params::Matrix{T}
    order::Bool
    VariationalCircuitRy(params::Matrix{T}, order=false) where T <: Number = new{T}(params, order)
    VariationalCircuitRy(N::Int, depth::Int; order=false, eltype=Float64) = new{eltype}(2π .* rand(N, depth), order)
    VariationalCircuitRy() = new{Float64}(Matrix{Float64}(undef, 0, 0), false) # Empty circuit to be used as a placeholder
end
Flux.@functor VariationalCircuitRy
Base.size(model::VariationalCircuitRy) = size(model.params)
Base.size(model::VariationalCircuitRy, i::Int) = size(model.params, i)
get_depth(model::VariationalCircuitRy) = size(model.params, 2)
get_N(model::VariationalCircuitRy) = size(model.params, 1)


function generate_circuit!(circuit, model::VariationalCircuitRy; params=nothing, N::Integer, depth::Integer)
    if model.order
        for d in 1:depth-1
            circuit = vcat(circuit, Rylayer(params[:, d]))
            circuit = vcat(circuit, CXlayer(N, d+1))
        end

        circuit = vcat(circuit, Rylayer(params[:, depth]))
    else
        for d in 1:depth
            circuit = vcat(circuit, CXlayer(N, d))
            circuit = vcat(circuit, Rylayer(params[:, d]))
        end
    end

    
    return circuit
end

VecVec = Vector{Vector{Float64}}
# Variational circuit with Ry gates
struct VariationalCircuitCorrRy <: AbstractVariationalCircuit
    params::Tuple
    odd::Bool
    VariationalCircuitCorrRy(params::Tuple, odd::Bool) = new(params, odd)
    VariationalCircuitCorrRy(params::Tuple; odd::Bool=false) = new(params, odd)
    VariationalCircuitCorrRy() = new(Matrix{Float64}(undef, 0, 0)) # Empty circuit to be used as a placeholder
end
Flux.@functor VariationalCircuitCorrRy
Base.size(model::VariationalCircuitCorrRy) = size(model.params)
get_depth(model::VariationalCircuitCorrRy) = size(model.params, 1)
get_N(model::VariationalCircuitCorrRy) = size(model.params[1], 1) + model.odd


function VariationalCircuitCorrRy(N::Integer, depth::Integer)
    params = Vector{Float64}[]
    if iseven(N)
        for i in 1:depth
            if iseven(i)
                push!(params, 2π .* rand(N-2))
            else
                push!(params, 2π .* rand(N))
            end
        end
        #push!(params, 2π .* rand(N)) # Last layer is complete
    else
        @assert false "odd circuits are not implemented correctly"
        for i in 1:depth
            push!(params, 2π .* rand(N-1))
        end
    end
    return VariationalCircuitCorrRy(Tuple(params), isodd(N))
end

function VariationalCircuitCorrRy(m::VariationalCircuitRy)
    
    params = Vector{Float64}[]
    N = get_N(m)
    depth = get_depth(m)
    for i in 1:depth
        if iseven(i)
            push!(params, m.params[2:N-1, i])
        else
            p = copy(m.params[:, i])
            if i != depth
                p[1] += m.params[1, i+1]
                p[end] += m.params[end, i+1]
            end
            push!(params, p)
        end
    end
    
    return VariationalCircuitCorrRy(Tuple(params))
end

function generate_circuit!(circuit, model::VariationalCircuitCorrRy; params=nothing, N::Integer, depth::Integer)
    if iseven(N)
        circuit = generate_circuit_even!(circuit, model; params, N, depth)
    else
        circuit = generate_circuit_odd!(circuit, model; params, N, depth)
    end
    return circuit
end


function generate_circuit_odd!(circuit, ::VariationalCircuitCorrRy; params=nothing, N::Integer, depth::Integer)
    # Throw not implemented errror
    @assert false "odd circuits are not implemented correctly"

    for d in 1:depth
        if iseven(d)
            @assert length(params[d]) == N-1 "Number of parameters must match number of qubits"
            circuit = vcat(circuit, Rylayer(params[d]))
        else
            @assert length(params[d]) == N-1 "Number of parameters must match number of qubits"
            circuit = vcat(circuit, Rylayer(params[d]; offset=1))
        end
    end
    return circuit
end

function generate_circuit_even!(circuit, ::VariationalCircuitCorrRy; params=nothing, N::Integer, depth::Integer)
    
    for d in 1:depth - 1
        if iseven(d)
            @assert length(params[d]) == N-2 "Number of parameters must match number of qubits"
            circuit = vcat(circuit, Rylayer(params[d]; offset=1))
        else
            @assert length(params[d]) == N "Number of parameters must match number of qubits"
            circuit = vcat(circuit, Rylayer(params[d]))
        end
        circuit = vcat(circuit, CXlayer(N, d+1))
    end
    if iseven(depth)
        circuit = vcat(circuit, Rylayer(params[depth]; offset=1))
    else
        circuit = vcat(circuit, Rylayer(params[depth]))
    end

    return circuit
end



# Variational circuit with Ry gates
struct VariationalCircuitSymRy <: AbstractVariationalCircuit
    params::Matrix{Float64}
    VariationalCircuitSymRy(params::Matrix{Float64}) = new(params)
    VariationalCircuitSymRy() = new(Matrix{Float64}(undef, 0, 0)) # Empty circuit to be used as a placeholder
end
Flux.@functor VariationalCircuitSymRy

Base.size(model::VariationalCircuitSymRy) = size(model.params)
Base.size(model::VariationalCircuitSymRy, i::Int) = size(model.params, i)
get_depth(model::VariationalCircuitSymRy) = size(model.params, 2) - 1
get_N(model::VariationalCircuitSymRy) = size(model.params, 1)


function VariationalCircuitSymRy(N::Integer, depth::Integer)
    @assert depth % 2 == 0 "depth must be even"
    depth_half = depth ÷ 2

    params_ = 2π .* rand(N, depth_half)
    params = zeros(N, depth + 1)

    params[:, 1:depth_half] .= params_
    params[:, depth_half+2:end] .= -params_[:, end:-1:1]
    params[:, end] .+= pi/2 # Last layer is not symmetric
    return VariationalCircuitSymRy(params)
end

function generate_circuit!(circuit, ::VariationalCircuitSymRy; params=nothing, N::Integer, depth::Integer)
    @assert depth % 2 == 0 "depth must be even"
    
    depth_half = depth ÷ 2

    for d in 1:depth_half
        circuit = vcat(circuit, Rylayer(params[:, d]))
        circuit = vcat(circuit, CXlayer(N, d))
    end

    circuit = vcat(circuit, Rylayer(params[:, depth_half+1]))

    for (d1, d2) in zip(depth_half:-1:1, depth_half+2:depth+1)
        circuit = vcat(circuit, CXlayer(N, d1))
        circuit = vcat(circuit, Rylayer(params[:, d2]))
    end

    return circuit
end


# Measurment circuit
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


include("VariationalCircuits/CompleteUnitary.jl")
include("FeedbackCircuits/FeedbackCircuits.jl")

#end
end # module