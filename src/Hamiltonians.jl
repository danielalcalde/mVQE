module Hamiltonians
using ITensors
using PastaQ

function hamiltonian_tfi(state_indices, h)
    os = OpSum()
    for (i, s) in enumerate(state_indices)
        os += -1, "Z", s, "Z", state_indices[mod1(i+1, length(state_indices))]
        os += -h, "X", s
    end
    
    return os
end

function hamiltonian_ghz(state_indices, hilbert)
    N = length(hilbert)
    state = zeros(Int, N)
    ψ_0 = productstate(hilbert, state)
    
    state[state_indices] .= 1
    ψ_1 = productstate(hilbert, state)
    ghz = (ψ_0 + ψ_1) / sqrt(2)
    return -outer(ghz, ghz')
end

end # module