using Pkg
using MethodAnalysis
println("Pkg status $(Pkg.status())")

@time begin
using LinearAlgebra
using ITensors
using PastaQ

end

println("precomp")
#=
@time begin
   # For loop over all files in precompile folder import and run them
   for file in readdir("precompile2")
        println("import $file")
        include("precompile2/$file")
       end
       for file in readdir("precompile2")
        println("precompiling $file")
        include("precompile2/$file")
        _precompile_()
       end
end
=#
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
using OptimizersExtension: OptimizerWrapper
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
