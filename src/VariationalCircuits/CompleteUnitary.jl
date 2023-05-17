abstract type AbstractVariationalCircuitBlock <: AbstractVariationalCircuit end
Base.size(model::AbstractVariationalCircuitBlock) = size(model.params)
Base.size(model::AbstractVariationalCircuitBlock, i::Int) = size(model.params, i)
get_depth(model::AbstractVariationalCircuitBlock) = size(model.params, 3)
get_N(model::AbstractVariationalCircuitBlock) = size(model.params, 1)
Flux.trainable(a::AbstractVariationalCircuitBlock) = (a.params,)

# Variational circuit with Ry, Rx, Rz and CRx gates
struct VariationalCircuitOverparametrizied <: AbstractVariationalCircuitBlock
    params::Array{Float64, 3}
    VariationalCircuitOverparametrizied(params::Array{Float64, 3}) = new(params)
    VariationalCircuitOverparametrizied(N::Int, depth::Int) = new(2π .* rand(N, 4, depth))
    VariationalCircuitOverparametrizied() = new(Array{Float64, 3}(undef, 0, 0, 0)) # Empty circuit to be used as a placeholder
end
Flux.@functor VariationalCircuitOverparametrizied 


function generate_circuit!(circuit, ::VariationalCircuitOverparametrizied; params=nothing, N::Integer, depth::Integer)
    for d in 1:depth
        circuit = vcat(circuit, Rxlayer(params[:, 1, d]))
        circuit = vcat(circuit, Rylayer(params[:, 2, d]))
        circuit = vcat(circuit, Rzlayer(params[:, 3, d]))
        circuit = vcat(circuit, CRxlayer(N, d, params[:, 4, d]))
    end
    return circuit
end


using ..Layers: Ulayer, CUlayer

# Variational circuit with U and CU gates
struct VariationalCircuitUnitary <: AbstractVariationalCircuitBlock
    params::Array{Float64, 3}
    VariationalCircuitUnitary(params::Array{Float64, 3}) = new(params)
    VariationalCircuitUnitary(N::Int, depth::Int) = new(2π .* rand(N, 7, depth))
    VariationalCircuitUnitary() = new(Array{Float64, 3}(undef, 0, 0, 0)) # Empty circuit to be used as a placeholder
end
Flux.@functor VariationalCircuitUnitary 

function generate_circuit!(circuit, ::VariationalCircuitUnitary; params=nothing, N::Integer, depth::Integer)
    for d in 1:depth
        circuit = vcat(circuit, Ulayer(params[:, 1:3, d]))
        circuit = vcat(circuit, CUlayer(N, d, params[:, 4:end, d]))
    end
    return circuit
end

using ..Layers: CUlayer_broken

# Variational circuit with U and CU gates
struct VariationalCircuitUnitaryBroken <: AbstractVariationalCircuitBlock
    params::Array{Float64, 3}
    broken::Int
    VariationalCircuitUnitaryBroken(params::Array{Float64, 3}; broken=6) = new(params, broken)
    VariationalCircuitUnitaryBroken(N::Int, depth::Int; broken=6) = new(2π .* rand(N, 7, depth), broken)
    VariationalCircuitUnitaryBroken(;broken=6) = new(Array{Float64, 3}(undef, 0, 0, 0), broken) # Empty circuit to be used as a placeholder
end
Flux.@functor VariationalCircuitUnitaryBroken 

function generate_circuit!(circuit, var::VariationalCircuitUnitaryBroken; params=nothing, N::Integer, depth::Integer)
    for d in 1:depth
        circuit = vcat(circuit, Ulayer(params[:, 1:3, d]))
        circuit = vcat(circuit, CUlayer_broken(N, d, params[:, 4:end, d]; broken=var.broken))
    end
    return circuit
end
