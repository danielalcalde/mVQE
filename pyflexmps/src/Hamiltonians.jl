module Hamiltonians
using ITensors
using PastaQ
using pyflexmps: convert_sympy_to_opsum, pfs
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


function translate_spinone_to_spinhalf_symb(i=0; spin_trans=Dict(), base="girvin")
    if base=="girvin"
        spin_trans[Sym(pfs.KetSpinOne("1", i))] = Sym(pfs.KetSpinHalf("-1/2", i)*pfs.KetSpinHalf("1/2", i+1))
        spin_trans[Sym(pfs.KetSpinOne("0", i))] = Sym(pfs.KetSpinHalf("1/2", i)*pfs.KetSpinHalf("1/2", i+1))
        spin_trans[Sym(pfs.KetSpinOne("-1", i))] = Sym(pfs.KetSpinHalf("1/2", i)*pfs.KetSpinHalf("-1/2", i+1))
    elseif base == "clebsch"
        spin_trans[Sym(pfs.KetSpinOne("1", i))] = Sym(pfs.KetSpinHalf.from_spin1(stot=1, sz=1, n1=i, n2=i+1))
        spin_trans[Sym(pfs.KetSpinOne("0", i))] = Sym(pfs.KetSpinHalf.from_spin1(stot=1, sz=0, n1=i, n2=i+1))
        spin_trans[Sym(pfs.KetSpinOne("-1", i))] = Sym(pfs.KetSpinHalf.from_spin1(stot=1, sz=-1, n1=i, n2=i+1))
    end
    return spin_trans
end

function hamiltonian_aklt_half_symb(;kwargs...)
    ham_aklt = 0
    spin_trans = translate_spinone_to_spinhalf_symb(0; kwargs...)
    spin_trans = translate_spinone_to_spinhalf_symb(2; spin_trans, kwargs...)
    for i in -2:2
        state = Sym(pfs.KetSpinOne.from_spin2(stot=2, sz=i, n1=0, n2=2))
        state = subs(state, spin_trans)

        ham_aklt += pfs.projector(state.o)
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
    return ham_spin1_girvin
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
    
    op1 = convert_sympy_to_opsum(ham_aklt, Vector(1:2:N_state-2); sublattice=sublattice)
    op2 = convert_sympy_to_opsum(ham_spin1, Vector(1:2:N_state); sublattice=sublattice)

    
    #return MPO(op1 + op2, hilbert_state), MPO(op1, hilbert_state), MPO(op2, hilbert_state)
    # Use invokelatest to avoid a worldline conflict with pyflexmps
    return Base.invokelatest(MPO, op1 + op2, hilbert_state), Base.invokelatest(MPO, op1, hilbert_state), Base.invokelatest(MPO, op2, hilbert_state)
end

# The Haldane phase
function get_projector_from_spinone_to_spinhalf(i)
    c = 0
    for (v, p) in translate_spinone_to_spinhalf_symb(i)
        c += sympy.adjoint(p)*v
    end
    return Sym(c)
end

function hamiltonian_haldane_half_symb(θ=atan(1/3); kwargs...)
    Sx1 = pfs.Spin1("x", 0)
    Sz1 = pfs.Spin1("z", 0)
    Sym_ = [0  1 0;
           -1 0 1;
           0 -1 0]
    
    Sy1 = -im * sympy.simplify(Sym(pfs.Spin1.from_matrix(Sym_, 0))/sympy.sqrt(2))
    Sx2 = pfs.Spin1("x", 2)
    Sz2 = pfs.Spin1("z", 2)
    Sy2 = -im * sympy.simplify(Sym(pfs.Spin1.from_matrix(Sym_, 2))/sympy.sqrt(2))

    S1 = [Sx1, Sy1, Sz1]
    S2 = [Sx2, Sy2, Sz2]
    
    # Project to  SpinHalf
    c0 = get_projector_from_spinone_to_spinhalf(0; kwargs...)
    c2 = get_projector_from_spinone_to_spinhalf(2; kwargs...)

    conv1 = S -> Sym(pfs.quantum_states.convert_ketbra_to_operator(pfs.apply((sympy.adjoint(c0)*Sym(S)*Sym(c0)).o)))
    S1c = conv1.(S1)

    conv2 = S -> Sym(pfs.quantum_states.convert_ketbra_to_operator(pfs.apply((sympy.adjoint(c2)*Sym(S)*Sym(c2)).o)))
    S2c = conv2.(S2)

    SS = sum(S1c .* S2c)
    SS2 = SS * SS
    id = Sym(pfs.sigmaid(0) * pfs.sigmaid(1) * pfs.sigmaid(2) * pfs.sigmaid(3))
    
    J = 0.5/cos(atan(1/3))
    J1, J2 = J .* (cos(θ), sin(θ))
    
    #H_term = 1/sympy.Number(2) * SS
    #H_term += 1/sympy.Number(6) * SS2

    H_term = J1 * SS
    H_term += J2 * SS2
    H_term += 1/sympy.Number(3) * id

    H_term = pfs.apply(H_term.o)
    return H_term
end

function hamiltonian_haldane_half(hilbert_state, θ=atan(1/3); sublattice=nothing, kwargs...)
    if sublattice === nothing
        N_state = length(hilbert_state)
        sublattice = Vector(1:N_state)
    else
        N_state = length(sublattice)
    end
    
    ham_haldane = hamiltonian_haldane_half_symb(θ; kwargs...)
    ham_spin1 = hamiltonian_aklt_spin1_symb(; kwargs...)
    
    op1 = convert_sympy_to_opsum(ham_haldane, Vector(1:2:N_state-2); sublattice=sublattice)
    op2 = convert_sympy_to_opsum(ham_spin1, Vector(1:2:N_state); sublattice=sublattice)
    
    #return MPO(op1 + op2, hilbert_state), MPO(op1, hilbert_state), MPO(op2, hilbert_state)
    # Use invokelatest to avoid a worldline conflict with pyflexmps
    return Base.invokelatest(MPO, op1 + op2, hilbert_state), Base.invokelatest(MPO, op1, hilbert_state), Base.invokelatest(MPO, op2, hilbert_state)
end

end # module