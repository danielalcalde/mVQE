function sample_and_probs(ρj::ITensor, s, d)
    # Compute the probability of each state
    # one-by-one and stop when the random
    # number r is below the total prob so far
    pdisc = 0.0
    r = rand()
    # Will need n, An, and pn below
    
    projn = ITensor()
    n = 1
    pn = 0.0
    while n <= d
        projn = ITensor(s)
        projn[s => n] = 1
        pnc = (ρj * projn * prime(projn))[]
        if imag(pnc) > 1e-6
            @warn "Sample probability $pnc is complex."
        end
        pn = real(pnc)
        pdisc += pn
        (r < pdisc) && break
        n += 1
    end
    return n, pn, projn
end


function projective_measurement_sample(ρ::MPO; indices=1:length(ρ), reset=nothing, norm_treshold=0.9)

    N = length(ρ)
    s = siteinds(ρ)
    R = Vector{ITensor}(undef, N)
    
    result = Vector{Int}(undef, length(indices))
    
    if reset === nothing || reset isa Int
        reset = fill(reset, length(indices))
    end
    
    ρ_tensors = ITensor[]

    Zygote.ignore() do
        R[N] = ρ[N] * δ(dag(s[N]))
        for n in reverse(1:(N - 1))
            R[n] = ρ[n] * δ(dag(s[n])) * R[n + 1]
        end

        norm_ = R[1][]
        if abs(1 - R[1][]) < norm_treshold
            R = R / norm_
            ρ[1] = ρ[1] / norm_
        else
            error("sample: MPO is not normalized, norm=$(tr(ρ)), $(abs(1 - R[1][]))> $norm_treshold")
        end
    
    end
    
    ρj = Zygote.@ignore ρ[1] * R[2]
    Lj = ITensor()
    i = 1
    prob = 0.
    projn = ITensor()
    for j in 1:N
        s = siteind(ρ, j)
        d = dim(s)
        
        if j in indices
            Zygote.@ignore result[i], prob, projn = sample_and_probs(ρj, s, d)
            gate = Zygote.@ignore projective_measurement_gate_sample(s, result[i], prob; reset=reset[i])
            ρ_j = product(gate, ρ[j]; apply_dag=true)
            
            Zygote.ignore() do
                if j < N
                    if j == 1
                        Lj = ρ[1] * projn * prime(projn)
                    elseif j > 1
                        Lj = Lj * ρ[j] * projn * prime(projn)
                    end
                    if j == N - 1
                        ρj = Lj * ρ[N]
                    else
                        ρj = Lj * ρ[j + 1] * R[j + 2]
                    end
                    s = siteind(ρ, j + 1)
                    normj = (ρj * δ(s', s))[]
                    ρj ./= normj
                end
                i += 1
            end
        else
            ρ_j = ρ[j]
            
            Zygote.ignore() do
            
                if j < N
                    if j == 1
                        Lj = ρ[1] * δ(s', s)
                    elseif j > 1
                        Lj = Lj * ρ[j] * δ(s', s)
                    end
                    if j == N - 1
                        ρj = Lj * ρ[N]
                    else
                        ρj = Lj * ρ[j + 1] * R[j + 2]
                    end
                    s = siteind(ρ, j + 1)
                    normj = (ρj * δ(s', s))[]
                    ρj ./= normj
                end
            end
        end
        ρ_tensors = vcat(ρ_tensors, ρ_j)
        
    end

    return MPO(ρ_tensors), result
end

function projective_measurement_sample2(ρ::MPO; indices=1:length(ρ), reset=nothing, norm_treshold=0.9)
    gates = Vector{ITensor}(undef, length(indices))
    result = Vector{Int}(undef, length(indices))

    Zygote.ignore() do
        println("ss")
        N = length(ρ)
        s = siteinds(ρ)
        R = Vector{ITensor}(undef, N)

        if reset === nothing || reset isa Int
            reset = fill(reset, length(indices))
        end

        
        R[N] = ρ[N] * δ(dag(s[N]))
        for n in reverse(1:(N - 1))
            R[n] = ρ[n] * δ(dag(s[n])) * R[n + 1]
        end

        # Normalize the MPO if the normalization is not to bad
        if abs(1 - R[1][]) < norm_treshold
            R = R / R[1][]
            ρ[1] = ρ[1] / R[1][]
        else
            error("sample: MPO is not normalized, norm=$(tr(ρ)), $(abs(1 - R[1][]))> $norm_treshold")
        end
        
        ρj = ρ[1] * R[2]
        Lj = ITensor()
        i = 1
        for j in 1:N
            s = siteind(ρ, j)
            d = dim(s)
            
            if j in indices
                result[i], prob, projn = sample_and_probs(ρj, s, d)
                gates[i] = projective_measurement_gate_sample(s, result[i], prob; reset=reset[i])

                if j < N
                    if j == 1
                        Lj = ρ[1] * projn * prime(projn)
                    elseif j > 1
                        Lj = Lj * ρ[j] * projn * prime(projn)
                    end
                    if j == N - 1
                        ρj = Lj * ρ[N]
                    else
                        ρj = Lj * ρ[j + 1] * R[j + 2]
                    end
                    s = siteind(ρ, j + 1)
                    normj = (ρj * δ(s', s))[]
                    ρj ./= normj
                end
                i += 1
            else
                
                if j < N
                    if j == 1
                        Lj = ρ[1] * δ(s', s)
                    elseif j > 1
                        Lj = Lj * ρ[j] * δ(s', s)
                    end
                    if j == N - 1
                        ρj = Lj * ρ[N]
                    else
                        ρj = Lj * ρ[j + 1] * R[j + 2]
                    end
                    s = siteind(ρ, j + 1)
                    normj = (ρj * δ(s', s))[]
                    ρj ./= normj
                end
            end
        end
    end

    return apply(gates, ρ; apply_dag=true), result
end
