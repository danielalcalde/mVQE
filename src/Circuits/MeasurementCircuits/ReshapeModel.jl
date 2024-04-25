mutable struct ReshapeModel
    model
    output_shape
    input_shape
    ReshapeModel(model, output_shape, input_shape=(:,)) = new(model, output_shape, input_shape)
end
Flux.@functor ReshapeModel
Flux.trainable(a::ReshapeModel) = (a.model,)

function (f::ReshapeModel)(input; batched=false, kwargs...)
    input_shape = f.input_shape
    output_shape = f.output_shape
    if batched
        batch_dim = size(input)[end]
        input_shape = (input_shape..., batch_dim)
        output_shape = (output_shape..., batch_dim)
    end
    
    input = reshape(input, input_shape)
    output = f.model(input; kwargs...)

    return reshape(output, output_shape)
end


struct NestedReshapeModel
    model
    output_shape::Tuple
end
Flux.@functor NestedReshapeModel
Flux.trainable(a::NestedReshapeModel) = (a.model,)

function (f::NestedReshapeModel)(input; kwargs...)
    out = f.model(input[:]; kwargs...)
    n = 1
    out_r = fmap(f.output_shape) do shape
        dim = prod(shape)
        r = reshape(out[n:n+dim-1], Tuple(shape))
        n += dim
        return r
    end
    return out_r
end