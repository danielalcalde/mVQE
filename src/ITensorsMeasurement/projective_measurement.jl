function projective_measurement_gate(s; reset=nothing)
    sₚ = prime(s)
    kraus = Index(s.space, "kraus")
    projn = ITensor(sₚ, s, kraus)
    Zygote.ignore() do
        if reset === nothing   
            for l in 1:s.space
                projn[sₚ => l, s => l, kraus => l] = 1
            end
        else
            for l in 1:s.space
                projn[sₚ => reset, s => l, kraus => l] = 1
            end
        end
    end
    return projn
end


function tr(ρ::MPO, indices)
    N = length(ρ)
    
    ρ_tensors = ITensor[]
    tracer = nothing
    j = 1
    for i in 1:N
        ρi = ρ[i]
        if i in indices
            if tracer === nothing
                tracer = tr(ρi)
            else
                tracer = contract(tracer, tr(ρi))
            end
            
            j += 1
        else
            if tracer === nothing
                ρ_tensors = vcat(ρ_tensors, ρi)
            else
                ρi = contract(tracer, ρi)
                ρ_tensors = vcat(ρ_tensors, ρi)
                tracer = nothing
            end
        end
    end

    if tracer !== nothing
        ρi = contract(tracer, ρ_tensors[end])
        ρ_tensors = vcat(ρ_tensors[1:end-1], ρi)
    end

    return MPO(ρ_tensors)
end

function projective_measurement(ρ::MPO; indices=1:length(ρ), reset=nothing)
    N = length(ρ)
    
    if reset !== nothing && reset isa Int
        reset = fill(reset, length(indices))
    end
    ρ_tensors = ITensor[]
    j = 1
    for i in 1:N
        ρi = ρ[i]
        if i in indices
            s1 = siteind(ρ, i)
            s2 = prime(s1)
            s1ₚ = prime(s2)
            s2ₚ = prime(s2')
            projn = ITensor(s1, s2, s1ₚ, s2ₚ)
            Zygote.ignore() do
                if reset === nothing   
                    for l in 1:s1.space
                        projn[s1 => l, s2 => l, s1ₚ => l, s2ₚ => l] = 1
                    end
                else
                    for l in 1:s1.space
                        projn[s1 => l, s2 => l, s1ₚ => reset[j], s2ₚ => reset[j]] = 1
                    end
                end
            end
            
            ρi = ρi * projn
            ρi = prime(ρi, -2, s1ₚ)
            ρi = prime(ρi, -2, s2ₚ)
            j += 1
        end
        ρ_tensors = vcat(ρ_tensors, ρi)
    end

    return MPO(ρ_tensors)
end


function projective_measurement(ψ::MPS; kwargs...)
    ρ = outer(ψ, ψ')
    ρ = projective_measurement(ρ; kwargs...)
    return ρ
end


function projective_measurement(ψs::Vector{MPS}; kwargs...) 
    ψs_out = MPS[]
    res_out = Vector{Vector{Int64}}()
    for ψ in ψs
        ψ = projective_measurement(ψ; kwargs...)
        ψs_out = vcat(ψs_out, [ψ])
    end

    return ψs_out
end

function projective_measurement(ρs::Vector{MPO}; kwargs...)
    ρs_out = MPO[]
    for ρ in ρs
        ρ = projective_measurement(ρ; kwargs...)[1]
        ρs_out = vcat(ρs_out, [ρ])
    end

    return ρs_out
end