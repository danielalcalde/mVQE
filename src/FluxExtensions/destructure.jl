function destructure(m; get_restructure_grads=false, ignore=[], original=nothing)
    xs = Zygote.Buffer([])
    
    if original === nothing
        params = Flux.params(m)
        fmap(m) do x
            x in params && !(x in ignore) && push!(xs, x)
            return x
        end
    else
        params = Flux.params(original)
        fmap(m, original) do x, xo
            xo in params && !(xo in ignore) && push!(xs, x)
            return x
        end
    end
    θ = vcat(vec.(copy(xs))...)
    re = p -> _restructure(m, p; ignore=ignore)

    if get_restructure_grads
        re_grads = m -> _restructure_grads(m, params; ignore=ignore)
        return θ, re, re_grads
    else
        return θ, re
    end
  end
  
  
  function _restructure(m, xs; ignore=[])
      i = 0
      params = Flux.params(m)
      fmap(m) do x
            (x in params && !(x in ignore) )|| return x
            x = reshape(xs[i.+(1:length(x))], size(x))
            i += length(x)
            return x
      end
  end
  
  Zygote.@adjoint function _restructure(m, xs; ignore=[])
    _restructure(m, xs; ignore=ignore), dm -> (nothing, destructure(dm; ignore=ignore, original=m)[1])
  end



function destructure_grads(g; ignore=[])
    params = g.params
    gs = [g[p] for p in params if !(p in ignore)]
    return vcat(vec.(copy(gs))...), m -> _restructure_grads(m, params; ignore=ignore)
end

function get_destructure_grads(g; ignore=[])
  params = g.params
  function destructure_grads(g)
    gs = [g[p] for p in params if !(p in ignore)]
    return vcat(vec.(copy(gs))...)
  end
  return destructure_grads, m -> _restructure_grads(m, params; ignore=ignore)
end

function _restructure_grads(m, params; ignore=ignore)
    i = 0
    g_new = IdDict()
    for param in params
        if param in ignore
            continue
        end
        g_new[param] = reshape(m[i.+(1:length(param))], size(param))
        i += length(param)
    end
    return Zygote.Grads(g_new, params)
end