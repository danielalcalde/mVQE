function nfan_std(param; mode=:glorot)
    dims = size(param)
    fan = Flux.nfan(dims...)
    if fan[1] == 1
        return 0.
    else
        if mode === :glorot
            return sqrt(2/sum(fan))
        elseif mode === :he
            return sqrt(2/fan[1])
        else
            error("Unknown mode: $mode")
        end
    end
end

function get_init_std(model; std_last_bias=0, destructure_model=false)
    model_std = deepcopy(model)
    for param in Flux.params(model_std)
        std = nfan_std(param)
        param .= std
    end
    if std_last_bias != 0
        model_std[end].bias .= std_last_bias
    end
    if destructure_model
        model_std = destructure(model_std)[1]
    end
    return model_std
end