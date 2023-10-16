# Variational circuit with a parametrized two qubit gate
struct VariationalCircuitTwoQubitGate <: AbstractVariationalCircuitBlock
    params::Array{Float64, 3}
    gate_type::String
    double_gate::Bool
    odd_first::Bool
    VariationalCircuitTwoQubitGate(params::Array{Float64, 3}; gate_type="CX_Id", double_gate=false, odd_first=false) = new(params, gate_type, double_gate, odd_first)
    function 
        VariationalCircuitTwoQubitGate(N::Int, depth::Int; gate_type="CX_Id", double_gate=false, odd_first=false)
        params = 2π .* rand(N, 2, depth) .- π
        params[:, 2, :] .= 2π
        return new(params, gate_type, double_gate, odd_first)
    end
    VariationalCircuitTwoQubitGate(; gate_type="CX_Id", double_gate=false, odd_first=false) = new(Array{Float64, 3}(undef, 0, 0, 0), gate_type, double_gate, odd_first) # Empty circuit to be used as a placeholder
end
Flux.@functor VariationalCircuitTwoQubitGate


function generate_circuit!(circuit, c::VariationalCircuitTwoQubitGate; params=nothing, N::Integer, depth::Integer)
    for d in 1:depth
        circuit = vcat(circuit, Rylayer(params[:, 1, d]))
        if d != depth
            circuit = vcat(circuit, BrickLayer(N, d + c.odd_first, params[:, 2, d]; gate=c.gate_type))
            if c.double_gate
                circuit = vcat(circuit, BrickLayer(N, d + c.odd_first + 1, params[N ÷ 2:end, 2, d]; gate=c.gate_type))
            end
        end
    end
    return circuit
end