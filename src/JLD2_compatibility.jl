function JLD2.rconvert(T::Type, x::JLD2.ReconstructedMutable{Symbol("mVQE.Circuits.VariationalCircuitCorrRy"), (:params, :odd), Tuple{Any, Bool}})
    return Circuits.VariationalCircuitCorrRy(x.params)
 end

function JLD2.rconvert(T::Type, x::JLD2.ReconstructedMutable{Symbol("mVQE.Circuits.VariationalCircuitRy{Float64}"), (:params), Tuple{Any}})
    return Circuits.VariationalCircuitRy(x.params)
end

function JLD2.rconvert(T::Type, x::JLD2.ReconstructedMutable{Symbol("mVQE.Circuits.VariationalCircuitRyPeriodic{Float64}"), (:params, :N), Tuple{Any, Int64}})
    return Circuits.VariationalCircuitRyPeriodic(x.params, x.N)
end

function JLD2.rconvert(T::Type, x::JLD2.ReconstructedMutable{Symbol("mVQE.Circuits.ReshapeModel"), (:model, :output_shape), Tuple{Any, Any}})
    return Circuits.ReshapeModel(x.model, x.output_shape)
end

function JLD2.rconvert(T::Type, x::JLD2.ReconstructedMutable{Symbol("mVQE.Circuits.VariationalCircuitRyPeriodic{Float64}"), (:params, :N, :holes_frequency), Tuple{Any, Int64, Int64}})
    return Circuits.VariationalCircuitRyPeriodic(x.params, x.N; holes_frequency=x.holes_frequency)
end