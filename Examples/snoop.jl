using SnoopCompile

### Log the compiles
# This only needs to be run once (to generate "/tmp/colortypes_compiles.log")
#=
SnoopCompile.@snoopc "./colortypes_compiles.log" begin
    using Pkg
    using MethodAnalysis
    println("Pkg status $(Pkg.status())")

    @time begin
    using LinearAlgebra
    using ITensors
    using PastaQ

    end
    println("import mVQE")
    @time begin
    import mVQE
    using mVQE.Measurements: projective_measurement
    using mVQE.StateFactory: random_MPS, infinite_temp_MPO
    using mVQE.Layers: Rylayer, CXlayer, Rxlayer
    using mVQE.Circuits: runcircuit, VariationalCircuitRy, VariationalMeasurement
    using mVQE: loss, optimize_and_evolve
    using mVQE.Hamiltonians: hamiltonian_tfi

    using mVQE.StateFactory: random_MPS, infinite_temp_MPO
    using mVQE.Layers: Rylayer, CXlayer, Rxlayer
    using mVQE.Circuits: runcircuit, VariationalCircuitRy, VariationalMeasurement, VariationalMeasurementMC
    using mVQE.Circuits: AbstractVariationalMeasurementCircuit, AbstractVariationalCircuit
    using mVQE: loss, optimize_and_evolve
    using mVQE.Misc: get_ancilla_indices, pprint
    using mVQE.Optimizers: OptimizerWrapper
    end

    println("comp")
    @time begin
    N_state = 3

    noise = (1 => ("depolarizing", (p = 0,)), 
            2 => ("depolarizing", (p = 0.0,)))

    # Define the ancilla indices
    state_indices, ancilla_indices, N = get_ancilla_indices(N_state, 1)

    # build MPO "cost function"
    hilbert = qubits(N)
    H = MPO(hamiltonian_tfi(state_indices, 0.5), hilbert)

    # Initialize state
    ψ = productstate(hilbert, fill(0, N))
    ρs = outer(ψ, ψ')
    end

end
=#
### Parse the compiles and generate precompilation scripts
# This can be run repeatedly to tweak the scripts
import Core
Core.String(e::Expr) = repr(e)

data = SnoopCompile.read("./colortypes_compiles.log")

pc = SnoopCompile.parcel(reverse!(data[2]))
SnoopCompile.write("./precompile", pc)