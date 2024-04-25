module mVQE

using ThreadsX
using Distributed
using ParallelDataTransfer

using Random
using Statistics

using OptimKit

using Zygote
import Flux

using ITensors
using PastaQ

using ITensorsExtensions

using ITensors: AbstractMPS
# Sub Modules
include("DistributedExtensions.jl")
include("FluxExtensions/FluxExtensions.jl")
include("Misc.jl")

include("ITensorsExtension.jl")
include("ITensorsMeasurement/measurement.jl")
include("MPOExtensions.jl")

include("StateFactory.jl")

include("Gates.jl")
include("Layers.jl")
include("Circuits/Circuits.jl")
include("GirvinProtocol.jl")


using ..ITensorsExtension: VectorAbstractMPS, States
using ..ITensorsMeasurement: projective_measurement, projective_measurement_sample
using ..Circuits: AbstractVariationalCircuit, AbstractVariationalMeasurementCircuit, generate_circuit
using OptimizersExtension: callback_, optimize

# Module
include("Optimization/Optimization.jl")
include("ParallelTools/ParallelTools.jl")

using JLD2
include("JLD2_compatibility.jl")

end