
struct VariationalCircuitRx{T <: Number} <: AbstractVariationalCircuit
    params::Matrix{T}
    order::Bool
    VariationalCircuitRx(params::Matrix{T}, order=false) where T <: Number = new{T}(params, order)
    VariationalCircuitRx(N::Int, depth::Int; order=false, eltype=Float64) = new{eltype}(2π .* rand(N, depth) .- π, order)
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
abstract type AbstractVariationalCircuitRy{T<:Number} <: AbstractVariationalCircuit end
struct VariationalCircuitRy{T <: Number} <: AbstractVariationalCircuitRy{T}
    params::Matrix{T}
    VariationalCircuitRy(params::Matrix{T},) where T <: Number = new{T}(params)
    VariationalCircuitRy(N::Int, depth::Int; eltype=Float64) = new{eltype}(2π .* rand(N, depth) .- π)
    VariationalCircuitRy() = new{Float64}(Matrix{Float64}(undef, 0, 0)) # Empty circuit to be used as a placeholder
end
Flux.@functor VariationalCircuitRy
Base.size(model::AbstractVariationalCircuitRy) = size(model.params)
Base.size(model::AbstractVariationalCircuitRy, i::Int) = size(model.params, i)
get_depth(model::AbstractVariationalCircuitRy) = size(model.params, 2)
get_N(model::AbstractVariationalCircuitRy) = size(model.params, 1)

function (model::AbstractVariationalCircuitRy{T})(ρ::States; params=nothing, eltype=nothing, kwargs...) where T <: Number
    params, N, depth = get_parameters(model; params=params)
    for d in 1:depth-1
        circ = Rylayer(params[:, d])
        ρ = ITensorsExtensions.runcircuit(ρ, circ; onequbit_gates=true, eltype=T, kwargs...)

        circ = CXlayer(N, d+1)
        ρ = ITensorsExtensions.runcircuit(ρ, circ; gate_grad=false, eltype=T, kwargs...)
    end
    circ = Rylayer(params[:, depth])
    ρ = ITensorsExtensions.runcircuit(ρ, circ; onequbit_gates=true, eltype=T, kwargs...)
    return ρ
end


struct VariationalCircuitRyPeriodic{T <: Number} <: AbstractVariationalCircuitRy{T}
    params::Matrix{T}
    N::Int
    VariationalCircuitRyPeriodic(params::Matrix{T}) where T <: Number = new{T}(params)
    VariationalCircuitRyPeriodic(N::Int, period::Int, depth::Int; eltype=Float64) = new{eltype}(2π .* rand(period, depth) .- π, N)
    VariationalCircuitRyPeriodic() = new{Float64}(Matrix{Float64}(undef, 0, 0), 0) # Empty circuit to be used as a placeholder
end
Flux.@functor VariationalCircuitRyPeriodic
get_N(model::VariationalCircuitRyPeriodic) = model.N

function get_parameters(model::VariationalCircuitRyPeriodic; params=nothing)
    if params === nothing
        @assert size(model.params, 1) > 0 "$(typeof(model)) is empty"
        params = model.params
    end

    N = get_N(model)
    depth = get_depth(model)

    # Fill in the parameters into N qubit circuit periodically
    duplications = div(N, size(params, 1))
    @assert duplications * size(params, 1) == N "Number of qubits must be a multiple of the number of parameters"

    params = repeat(params, duplications, 1)

    return params, N, depth
end

VecVec = Vector{Vector{Float64}}
# Variational circuit with Ry gates
struct VariationalCircuitCorrRy <: AbstractVariationalCircuit
    params::Tuple
    odd::Bool
    VariationalCircuitCorrRy(params::Tuple, odd::Bool) = new(params, odd)
    VariationalCircuitCorrRy(params::Tuple; odd::Bool=false) = new(params, odd)
    VariationalCircuitCorrRy() = new(()) # Empty circuit to be used as a placeholder
end
Flux.@functor VariationalCircuitCorrRy
vector_size(model::VariationalCircuitCorrRy) = fmap(x->collect(size(x)), model.params)
get_depth(model::VariationalCircuitCorrRy) = size(model.params, 1)
get_N(model::VariationalCircuitCorrRy) = size(model.params[1], 1) + model.odd


function VariationalCircuitCorrRy(N::Integer, depth::Integer)
    params = Vector{Float64}[]
    if iseven(N)
        for i in 1:depth
            if iseven(i)
                push!(params, 2π .* rand(N-2) .- π)
            else
                push!(params, 2π .* rand(N) .- π)
            end
        end
        #push!(params, 2π .* rand(N)) # Last layer is complete
    else
        @assert false "odd circuits are not implemented correctly"
        for _ in 1:depth
            push!(params, 2π .* rand(N-1) .- π)
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
