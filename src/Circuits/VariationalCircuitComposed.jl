
struct VariationalCircuitComposed <: AbstractVariationalCircuit
    circuits::Vector{AbstractVariationalCircuit}
    function VariationalCircuitComposed(circuits::Vector{T}) where T <: AbstractVariationalCircuit
        #for circuit in circuits
        #    @assert get_N(circuit) == get_N(circuits[1]) "All circuits must have the same number of qubits"
        #end
        new(circuits)
    end
end
Flux.@functor VariationalCircuitComposed
Base.size(model::VariationalCircuitComposed) = collect(size(model.circuits[i]) for i in 1:length(model.circuits))
Base.size(model::VariationalCircuitComposed, i::Int) = size(model.circuits[i])
get_N(model::VariationalCircuitComposed) = get_N(model.circuits[1])
get_depth(model::VariationalCircuitComposed) = sum(get_depth(circ) for circ in model.circuits)
Flux.trainable(model::VariationalCircuitComposed) = (circ for circ in model.circuits)

function (model::VariationalCircuitComposed)(ρ::States; params=nothing, eltype=nothing, kwargs...)
    if params === nothing
        params = [nothing for _ in model.circuits]
    else
        @assert length(params) == length(model.circuits) "Number of parameters must match number of circuits"
    end

    for (i, circuit) in enumerate(model.circuits)
        ρ = circuit(ρ; params=params[i], eltype=eltype, kwargs...)
    end
    return ρ
end

abstract type AbstractVariationalLayer <: AbstractVariationalCircuit end
Base.size(model::AbstractVariationalLayer) = size(model.params)
Base.size(model::AbstractVariationalLayer, i::Int) = size(model.params, i)
get_N(model::AbstractVariationalLayer) = length(model.sites)
get_depth(model::AbstractVariationalLayer) = 1

struct VariationalOneQubit{T <: Number} <: AbstractVariationalLayer
    sites::Vector{<:Integer}
    params::Vector{T}
    gate_type::String
end
Flux.@functor VariationalOneQubit

function VariationalOneQubit(N::Integer; gate_type="Ry", sites=1:N)
    @assert N == length(sites) "Number of sites must match number of qubits"
    params = 2π .* rand(N) .- π
    return VariationalOneQubit(collect(sites), params, gate_type)
end

function generate_circuit!(circuit, model::VariationalOneQubit; params=nothing, N::Integer, depth::Integer)
    params = select_params(params, model.params)
    return vcat(circuit, OneGateLayer(params; sites=model.sites, gate=model.gate_type))
end

struct VariationalOneQubitM{T <: Number} <: AbstractVariationalLayer
    sites::Vector{<:Integer}
    params::Matrix{T}
    gate_type::String
end
Flux.@functor VariationalOneQubitM

function VariationalOneQubitM(N::Integer; gate_type="U", sites=1:N, nr_params=1)
    @assert N == length(sites) "Number of sites must match number of qubits"
    params = 2π .* rand(N, nr_params) .- π
    return VariationalOneQubitM(collect(sites), params, gate_type)
end

function generate_circuit!(circuit, model::VariationalOneQubitM; params=nothing, N::Integer, depth::Integer)
    params = select_params(params, model.params)
    return vcat(circuit, OneGateLayer(params; sites=model.sites, gate=model.gate_type))
end

struct VariationalTwoQubit{T <: Number} <: AbstractVariationalLayer
    sites::Vector{<:Integer}
    params::Vector{T}
    gate_type::String
end

function VariationalTwoQubit(N::Integer; gate_type="CX_Id", sites=1:N)
    @assert N == length(sites) "Number of sites must match number of qubits"
    @assert N % 2 == 0 "Number of qubits must be even"
    params = 2π .* rand(N÷2) .- π
    return VariationalTwoQubit(collect(sites), params, gate_type)
end

function generate_circuit!(circuit, model::VariationalTwoQubit; params=nothing, N::Integer, depth::Integer)
    params = select_params(params, model.params)
    return vcat(circuit, BrickLayer(N, 1, params; gate=model.gate_type, sites=model.sites))
end