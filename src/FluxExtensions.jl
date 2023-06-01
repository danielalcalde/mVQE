module FluxExtensions
using Flux
using Zygote

struct ReshapeModel
    output_shape
end
Flux.@functor ReshapeModel
Flux.trainable(::ReshapeModel) = ()

(f::ReshapeModel)(input; kwargs...) = reshape(input, f.output_shape)


function ExpDecayGen(lr_start::Real, lr_end::Real, steps::Integer; decay_step=1, clip=lr_end)
    eff_steps = steps / decay_step
    decay = (lr_end/lr_start) ^ (1/eff_steps)
    return ExpDecay(lr_start, decay, decay_step, clip)
end

function destructure(m; get_restructure_grads=false)
    xs = Zygote.Buffer([])
    params = Flux.params(m)
    fmap(m) do x
      x in params && push!(xs, x)
      return x
    end
    θ = vcat(vec.(copy(xs))...)
    re = p -> _restructure(m, p)

    if get_restructure_grads
        re_grads = m -> _restructure_grads(m, params)
        return θ, re, re_grads
    else
        return θ, re
    end
  end
  
  
  function _restructure(m, xs)
    i = 0
    params = Flux.params(m)
    fmap(m) do x
      x in params  || return x
      x = reshape(xs[i.+(1:length(x))], size(x))
      i += length(x)
      return x
    end
  end
  
  Zygote.@adjoint function _restructure(m, xs)
    _restructure(m, xs), dm -> (nothing,destructure(dm)[1])
  end



function destructure_grads(g)
    params = g.params
    gs = [g[p] for p in params]
    return vcat(vec.(copy(gs))...), m -> _restructure_grads(m, params)
end

function get_destructure_grads(g)
  params = g.params
  function destructure_grads(g)
    gs = [g[p] for p in params]
    return vcat(vec.(copy(gs))...)
  end
  return destructure_grads, m -> _restructure_grads(m, params)
end

function _restructure_grads(m, params)
    i = 0
    g_new = IdDict()
    for param in params
        g_new[param] = reshape(m[i.+(1:length(param))], size(param))
        i += length(param)
    end
    return Zygote.Grads(g_new, params)
end
end # module

