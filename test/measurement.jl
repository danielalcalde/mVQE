N = 4

hilbert = qubits(N)
ψs = randomMPS(hilbert; linkdims=10)
ancilla_indices = [1, 3]
H = MPO(pyflexmps.Hamiltonians.hamiltonian_tfi(1:N, 0.5), hilbert)

@testset "projective_measurement_sample_psi" begin
    function f1(ψ)
        ψm, _ = mVQE.ITensorsMeasurement.projective_measurement_sample(ψ; indices=ancilla_indices)
        return inner(ψm', H, ψm)
    end
    
    function f2(ψ)
        ψm, _ = mVQE.ITensorsMeasurement.projective_measurement_sample_old(ψ; indices=ancilla_indices)
        return inner(ψm', H, ψm)
    end

    Random.seed!(1)
    E1, pull = pullback(f1, ψs)

    Random.seed!(1)
    E2, pull2 = pullback(f2, ψs)

    ψ1 = pull(1.)[1]
    ψ2 = pull2(1.)[1];

    @test E1 ≈ E2
    @test inner(ψ1,ψ1) ≈ inner(ψ2,ψ2)
    @test inner(ψ1,ψ2)/inner(ψ1,ψ1) ≈ 1.0
end

@testset "projective_measurement_sample_loglike" begin
    function f1(ψ)
        ψm, samples, log_prob = mVQE.ITensorsMeasurement.projective_measurement_sample(ψ; indices=ancilla_indices, get_loglike=true)
        return log_prob
    end
    Random.seed!(1)
    _, _, _, proj = mVQE.ITensorsMeasurement.projective_measurement_sample(ψs; indices=ancilla_indices, get_projectors=true, get_loglike=true);


    Random.seed!(1)
    log_like_1, pull1 = pullback(f1, ψs)

    proj_n = [proji/norm(proji) for proji in proj];
    function f2(ψ)
        ψs2 = apply(proj_n, ψ)
        return log(inner(ψs2, ψs2))
    end

    Random.seed!(1)
    log_like_2, pull2 = pullback(f2, ψs)


    psi1 = pull1(1)[1]
    psi2 = pull2(1)[1];

    @test log_like_1 ≈ log_like_2
    @test inner(psi1,psi1) ≈ inner(psi2,psi2)
    @test inner(psi1,psi2)/inner(psi1,psi1) ≈ 1.0
    
end