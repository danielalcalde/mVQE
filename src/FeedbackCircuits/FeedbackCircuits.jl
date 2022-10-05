abstract type AbstractFeedbackModel end

mutable struct VariationalMeasurementMCFeedback <: AbstractVariationalMeasurementCircuit
    vcircuit:: AbstractVariationalCircuit
    feedback_model:: AbstractFeedbackModel
    depth:: Int
    measurement_indices:: Vector{Int}
    reset:: Int
end

function Circuits.initialize_circuit(model::VariationalMeasurementMCFeedback)
    params = []
    θ = initialize_circuit(model.vcircuit)
    output_shape = size(θ)
    
    push!(params, θ)
    
    N_measurements = length(model.measurement_indices)
    for i in 2:model.depth
        input_shape = (i-1, N_measurements)
        θ = initialize_model(model.feedback_model, input_shape, output_shape)
        push!(params, θ)
    end
    return params
end


function PastaQ.runcircuit(ρ::States, model::VariationalMeasurementMCFeedback, θs; kwargs...)
    @assert length(θs) == model.depth
    measurements = Matrix{Int16}(undef, length(model.measurement_indices), model.depth)
    
    ρ = runcircuit(ρ, model.vcircuit, θs[1]; kwargs...)
    ρ, m = projective_measurement_sample(ρ; indices=model.measurement_indices, reset=model.reset)
    Zygote.@ignore measurements[:, 1] = m
    
    for i in 2:model.depth
        θ = run_model(measurements[:, 1:i-1], model.feedback_model, θs[i])
        ρ = runcircuit(ρ, model.vcircuit, θ; kwargs...)
        ρ, m = projective_measurement_sample(ρ; indices=model.measurement_indices, reset=model.reset)
        Zygote.@ignore measurements[:, i] = m
    end

    return ρ
end

include("LinearFeedback.jl")