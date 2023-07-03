function get_enviroments(ψ1, ψ2, right::Bool=false)
    @assert length(ψ1) == length(ψ2)
    @assert !any(linkinds(ψ1) .== linkinds(ψ2)) "ψ1 and ψ2 have common links"
    if right
        ψ1 = ψ1[end:-1:1]
        ψ2 = ψ2[end:-1:1]
    end
    
    envs = Vector{ITensor}(undef, length(ψ1))
    envs[1] = ψ1[1]*ψ2[1]
    
    for (i, (ψ1i, ψ2i)) in enumerate(zip(ψ1[2:end], ψ2[2:end]))
        envs[i + 1] = envs[i] * ψ1i * ψ2i
    end
    
    if right
        envs = envs[end:-1:1]
    end
    
    return envs
end