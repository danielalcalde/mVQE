struct FeedbackCircuit <: AbstractVariationalCircuit
    vcircuit::AbstractVariationalCircuit
    model
end
Flux.@functor FeedbackCircuit
Flux.trainable(a::FeedbackCircuit) = (a.model,)
Base.show(io::IO, model::FeedbackCircuit) = print(io, "FeedbackCircuit(\n ", model.vcircuit, ",\n ", model.model, ")")

function FeedbackCircuit(vcircuit, model, input_shape; size_independent=false)
    output_shape = vector_size(vcircuit)
    
    input_dims = prod(input_shape)

    if output_shape isa Vector
        output_dims = prod(output_shape)
        model = model(input_dims, output_dims)
        if size_independent
            # assume first dimension is the system size
            output_shape = (:, output_shape[2:end]...)
        end
        model = ReshapeModel(model, Tuple(output_shape))

    elseif output_shape isa Tuple # Nested tuple
        output_dims = 1
        fmap(output_shape) do shape
            output_dims *= prod(shape)
        end
        model = model(input_dims, output_dims)
        model = NestedReshapeModel(model, output_shape)
    end

    
    return FeedbackCircuit(vcircuit, model)
end

function (model::FeedbackCircuit)(ψ::States, measurement::Matrix{T}; kwargs...) where T <: Real
    params = model.model(measurement)
    ψ = model.vcircuit(ψ; params=params, kwargs...)
    return ψ
end

struct VariationalMeasurementMCFeedback <: AbstractVariationalMeasurementCircuit
    vcircuits:: Vector{AbstractVariationalCircuit}
    measurement_indices::Vector{<:Integer}
    reset::Union{Nothing, Integer}
end
Flux.@functor VariationalMeasurementMCFeedback

"""
    VariationalMeasurementMCFeedback(vcircuits, feedback_models, measurement_indices; reset=1)

Construct a variational measurement circuit with feedback.

Arguments:
- `vcircuits`: List of Variational Circuits
- `feedback_models`: List of functions that take the input shape and output shape and return a Flux model
- `measurement_indices`: List of indices to measure
- `reset`: measurement_indices will be reset to |reset> after measurement

Examples:
```julia
using Flux
using Flux: Dense
using mVQE

N = 4
depth = 2
num_measurements = 2
num_circuits = 3

vcircuits = [VariationalCircuitRy(N, depth) for _ in 1:num_circuits]
feedback_models = [Dense for _ in 2:num_circuits]
measurement_indices = [1, 2]

model = VariationalMeasurementMCFeedback(vcircuits, feedback_models, measurement_indices)
```
"""
function VariationalMeasurementMCFeedback(vcircuits::Vector{T}, feedback_models::Vector, measurement_indices:: Vector{<:Integer};
                                          reset::Union{Nothing, Integer}=1, size_independent=false, markovian=false) where T <: AbstractVariationalCircuit
    # Initialize feedback models
    N_measurements = length(measurement_indices)
    @assert length(feedback_models) == length(vcircuits) - 1

    vcircuits_new = Vector{AbstractVariationalCircuit}(undef, length(vcircuits))
    vcircuits_new[1] = vcircuits[1]

    for (i, model) in enumerate(feedback_models)
        if markovian
            input_shape = (1, N_measurements,)
        else
            input_shape = (i, N_measurements)
        end
        vcircuits_new[i+1] = FeedbackCircuit(vcircuits[i + 1], model, input_shape; size_independent)
    end

    return VariationalMeasurementMCFeedback(vcircuits_new, measurement_indices, reset)
end


function (model::VariationalMeasurementMCFeedback)(ρ::AbstractMPS;
    get_loglike=false, get_measurements=false, gradient_averaging=true, fake_measurement_feedback=false, kwargs...)

    measurements = Matrix{Int16}(undef, length(model.measurement_indices), length(model))
    ρ = model.vcircuits[1](ρ; kwargs...)
    local m, loglike
    if !fake_measurement_feedback
        ρ, m, loglike = projective_measurement_sample(ρ; indices=model.measurement_indices, reset=model.reset, get_loglike=true, gradient_averaging)
    else
        _, m, _ =       projective_measurement_sample(ρ; indices=model.measurement_indices, reset=model.reset, get_loglike=true, gradient_averaging)
        ρ, _, loglike = projective_measurement_sample(ρ; indices=model.measurement_indices, reset=model.reset, get_loglike=true, gradient_averaging)
    end
    
    Zygote.@ignore measurements[:, 1] = m .- 1

    for (i, vcircuit) in enumerate(model.vcircuits[2:end])
        eltype_ = real(Base.eltype(ρ[1]))
        M = Zygote.@ignore eltype_.(measurements[:, 1:i])
        ρ = vcircuit(ρ, M; kwargs...)
        
        local loglike_
        if !fake_measurement_feedback
            ρ, m, loglike_ = projective_measurement_sample(ρ; indices=model.measurement_indices, reset=model.reset, get_loglike=true, gradient_averaging)
        else
            _, m, _ =        projective_measurement_sample(ρ; indices=model.measurement_indices, reset=model.reset, get_loglike=true, gradient_averaging)
            ρ, _, loglike_ = projective_measurement_sample(ρ; indices=model.measurement_indices, reset=model.reset, get_loglike=true, gradient_averaging)
        end
        
        loglike += loglike_
        Zygote.@ignore measurements[:, i+1] = m .- 1
    end

    if get_measurements
        if get_loglike
            return ρ, measurements, loglike
        else
            return ρ, measurements
        end
    else
        if get_loglike
            return ρ, loglike
        else
            return ρ
        end
    end
end