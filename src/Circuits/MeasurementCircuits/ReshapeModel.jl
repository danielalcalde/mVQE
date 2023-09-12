struct ReshapeModel
    model
    output_shape
end
Flux.@functor ReshapeModel
Flux.trainable(a::ReshapeModel) = (a.model,)

function (f::ReshapeModel)(input; kwargs...)
    o = f.model(input[:]; kwargs...)
    @assert length(o) == prod(f.output_shape) "ReshapeModel: Output shape $(size(o)) does not match $(f.output_shape)"
    return reshape(o, f.output_shape)
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