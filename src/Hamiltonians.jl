module Hamiltonians
using ITensors
using PastaQ
using ..pyflexmps: convert_sympy_to_opsum, pfs
using SymPy

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


function translate_spinone_to_spinhalf_symb(;base="girvin")
    spin_trans = Dict()
    if base=="girvin"
        for i in [0, 2]
            spin_trans[pfs.KetSpinOne("1", i)] = pfs.KetSpinHalf("-1/2", i)*pfs.KetSpinHalf("1/2", i+1)
            spin_trans[pfs.KetSpinOne("0", i)] = pfs.KetSpinHalf("1/2", i)*pfs.KetSpinHalf("1/2", i+1)
            spin_trans[pfs.KetSpinOne("-1", i)] = pfs.KetSpinHalf("1/2", i)*pfs.KetSpinHalf("-1/2", i+1)
        end
    elseif base == "clebsch"
        for i in [0, 2]
            spin_trans[pfs.KetSpinOne("1", i)] = pfs.KetSpinHalf.from_spin1(stot=1, sz=1, n1=i, n2=i+1)
            spin_trans[pfs.KetSpinOne("0", i)] = pfs.KetSpinHalf.from_spin1(stot=1, sz=0, n1=i, n2=i+1)
            spin_trans[pfs.KetSpinOne("-1", i)] = pfs.KetSpinHalf.from_spin1(stot=1, sz=-1, n1=i, n2=i+1)
        end
    end
    return spin_trans
end

function hamiltonian_aklt_half_symb(;kwargs...)
    ham_aklt = 0
    spin_trans = translate_spinone_to_spinhalf_symb(;kwargs...)
    for i in -2:2
        state = pfs.KetSpinOne.from_spin2(stot=2, sz=i, n1=0, n2=2)
        state = subs(state, spin_trans)
        ham_aklt += pfs.projector(state)
    end
    return ham_aklt
end

function hamiltonian_aklt_spin1_symb(;base="girvin")
    if base == "girvin"
        singlet = pfs.KetSpinHalf("-1/2", 0) * pfs.KetSpinHalf("-1/2", 1)
    elseif base == "clebsch"
        singlet = pfs.KetSpinHalf.from_spin1(stot=0, sz=0, n1=0, n2=1)
    end
    ham_spin1_girvin = pfs.projector(singlet)
end

function hamiltonian_aklt_half(hilbert_state; sublattice=nothing, kwargs...)
    if sublattice === nothing
        N_state = length(hilbert_state)
        sublattice = Vector(1:N_state)
    else
        N_state = length(sublattice)
    end
    

    ham_aklt = hamiltonian_aklt_half_symb(;kwargs...)
    ham_spin1 = hamiltonian_aklt_spin1_symb(;kwargs...)

    sites = Vector(1:2:N_state-2)
    op1 = convert_sympy_to_opsum(ham_aklt, sites; sublattice=sublattice)

    sites = Vector(1:2:N_state)
    op2 = convert_sympy_to_opsum(ham_spin1, sites; sublattice=sublattice)
    return MPO(op1 + op2, hilbert_state), MPO(op1, hilbert_state), MPO(op2, hilbert_state)
end

end # module