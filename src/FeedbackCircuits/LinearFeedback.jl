mutable struct LinearFeedbackModel <: AbstractFeedbackModel
    output_shape
    LinearFeedbackModel() = new(nothing)
end

function initialize_model(model::LinearFeedbackModel, input_shape, output_shape)
    input_size = prod(input_shape)
    output_size = prod(output_shape)

    model.output_shape = output_shape

    W = Flux.glorot_uniform(output_size, input_size)
    b = zeros(output_size)
    return [W, b]
end

function run_model(input, model::LinearFeedbackModel, θ)
    res =  θ[1] * input[:] .+ θ[2]
    return reshape(res, model.output_shape)
end