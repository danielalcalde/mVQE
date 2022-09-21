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



end # module