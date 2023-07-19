
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