# Variational circuit with a parametrized two qubit gate
struct VariationalCircuitTwoQubitGate <: AbstractVariationalCircuitBlock
    params::Array{Float64, 3}
    gate_type::String
    double_gate::Bool
    odd_first::Bool
    sites::Vector{<:Integer}
    VariationalCircuitTwoQubitGate(params::Array{Float64, 3}; gate_type="CX_Id", double_gate=false, odd_first=false, sites=collect(1:size(params, 1))) = new(params, gate_type, double_gate, odd_first, sites)
    function VariationalCircuitTwoQubitGate(N::Int, depth::Int; gate_type="CX_Id", double_gate=false, odd_first=false, sites=collect(1:N))
        params = 2π .* rand(N, 2, depth) .- π
        params[:, 2, :] .= 2π
        return new(params, gate_type, double_gate, odd_first, sites)
    end
    VariationalCircuitTwoQubitGate(; gate_type="CX_Id", double_gate=false, odd_first=false, sites=collect(1:size(params, 1))) = new(Array{Float64, 3}(undef, 0, 0, 0), gate_type, double_gate, odd_first, sites) # Empty circuit to be used as a placeholder
end
Flux.@functor VariationalCircuitTwoQubitGate


function generate_circuit!(circuit, v::VariationalCircuitTwoQubitGate; params=nothing, N::Integer, depth::Integer)
    for d in 1:depth
        circuit = vcat(circuit, Rylayer(params[:, 1, d]))
        if d != depth
            circuit = vcat(circuit, BrickLayer(N, d + v.odd_first, params[:, 2, d]; gate=v.gate_type, sites=v.sites))
            if v.double_gate
                circuit = vcat(circuit, BrickLayer(N, d + v.odd_first + 1, params[N ÷ 2:end, 2, d]; gate=v.gate_type, sites=v.sites))
            end
        end
    end
    return circuit
end


mutable struct VariationalCircuitTwoQubitGatePeriodic <: AbstractVariationalCircuitBlock
    N::Int
    params::Array{Float64, 3}
    gate_type::String
    double_gate::Bool
    odd_first::Bool
    sites::Vector{<:Integer}
    VariationalCircuitTwoQubitGatePeriodic(N, params::Array{Float64, 3}; gate_type="CX_Id", double_gate=false, odd_first=false, sites=collect(1:N)) = new(N, params, gate_type, double_gate, odd_first, sites)
    function VariationalCircuitTwoQubitGatePeriodic(N::Int, period::Int, depth::Int; gate_type="CX_Id", double_gate=false, odd_first=false, sites=collect(1:N))
        @assert mod(period, 2) == 0
        @assert mod(N, period) == 0 
        params = 2π .* rand(period÷2, 3, depth) .- π
        params[:, 3, :] .= 2π
        return new(N, params, gate_type, double_gate, odd_first, sites)
    end
    VariationalCircuitTwoQubitGate(N; gate_type="CX_Id", double_gate=false, odd_first=false, sites=collect(1:N)) = new(N, Array{Float64, 3}(undef, 0, 0, 0), gate_type, double_gate, odd_first, sites) # Empty circuit to be used as a placeholder
end
Flux.@functor VariationalCircuitTwoQubitGatePeriodic
get_N(model::VariationalCircuitTwoQubitGatePeriodic) = model.N

function VariationalCircuitTwoQubitGatePeriodic(model::VariationalCircuitRyPeriodic)
    period = size(model.params, 1)
    depth = size(model.params, 2)
    params_ = reshape(model.params, 2, period÷2, depth)
    params = zeros(period÷2, 3, depth)
    params[:, 1:2, :] = permutedims(params_, (2, 1, 3))
    params[:, 3, :] .= 2π
    N = get_N(model)
    return VariationalCircuitTwoQubitGatePeriodic(N, params; odd_first=true)
end

function get_parameters(model::VariationalCircuitTwoQubitGatePeriodic; params=nothing)
    if params === nothing
        @assert size(model.params, 1) > 0 "$(typeof(model)) is empty"
        params = model.params
    end
    size_ = size(params)
    N = get_N(model)
    depth = size_[3]
    # Fill in the parameters into N qubit circuit periodically
    period = size(params, 1) * 2
    duplications = div(N, period)
    @assert duplications * period == N "Number of qubits must be a multiple of the number of parameters"
    params = reshape(params, 1, size_[1], size_[2], size_[3])
    params = repeat(params, 1, duplications, 1, 1)
    params = reshape(params, N÷2, size_[2], size_[3])
    
    return params, N, depth
end


function generate_circuit!(circuit, v::VariationalCircuitTwoQubitGatePeriodic; params=nothing, N::Integer, depth::Integer)
    for d in 1:depth
        p = transpose(params[:, 1:2, d])
        circuit = vcat(circuit, Rylayer(p[:]))
        if d != depth
            circuit = vcat(circuit, BrickLayer(N, d + v.odd_first, params[:, 3, d]; gate=v.gate_type, sites=v.sites))
        end
    end
    return circuit
end