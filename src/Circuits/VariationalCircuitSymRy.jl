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