struct ReshapeModel
    model
    output_shape
end
Flux.@functor ReshapeModel
Flux.trainable(a::ReshapeModel) = (a.model,)

(f::ReshapeModel)(input; kwargs...) = reshape(f.model(input[:]; kwargs...), f.output_shape)


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