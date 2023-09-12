# Variational circuit with a parametrized two qubit gate
struct VariationalCircuitTwoQubitGate <: AbstractVariationalCircuitBlock
    params::Array{Float64, 3}
    gate_type::String
    double_gate::Bool
    VariationalCircuitTwoQubitGate(params::Array{Float64, 3}; gate_type="CX_Id", double_gate=false) = new(params, gate_type, double_gate)
    VariationalCircuitTwoQubitGate(N::Int, depth::Int; gate_type="CX_Id", double_gate=false) = new(2π .* rand(N, 2, depth), gate_type, double_gate)
    VariationalCircuitTwoQubitGate(; gate_type="CX_Id", double_gate=false) = new(Array{Float64, 3}(undef, 0, 0, 0), gate_type, double_gate) # Empty circuit to be used as a placeholder
end
Flux.@functor VariationalCircuitTwoQubitGate


function generate_circuit!(circuit, c::VariationalCircuitTwoQubitGate; params=nothing, N::Integer, depth::Integer)
    for d in 1:depth
        circuit = vcat(circuit, Rylayer(params[:, 1, d]))
        circuit = vcat(circuit, BrickLayer(N, d, params[:, 2, d]; gate=c.gate_type))
        if c.double_gate
            circuit = vcat(circuit, BrickLayer(N, d, params[N ÷ 2:end, 2, d+1]; gate=c.gate_type))
        end
    end
    return circuit
end