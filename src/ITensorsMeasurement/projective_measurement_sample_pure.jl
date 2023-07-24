function sample_and_probs_mps(A::ITensor, s, d; random_number=rand())
    local An, projn, pn

    pdisc = 0.
    n = 1
    while n <= d
        projn = ITensor(s)
        projn[s => n] = 1
        An = A * dag(projn)
        pn = real(scalar(dag(An) * An))
        pdisc += pn
        (random_number < pdisc) && break
        n += 1
    end
    return n, pn, An
end

function sample_and_probs_mps2(P::ITensor, ψi::ITensor, s, linkind_P, d; random_number=rand())
    local An, projn, prob, Pn

    pdisc = 0
    n = 1
    # Remove the prime from the index A.tensor.inds[1]
    if linkind_P !== nothing
        tracer = δ(eltype(ψi), linkind_P, prime(linkind_P))
    end
    
    while n <= d
        projn = ITensor(s)
        projn[s => n] = 1
        An = ψi * dag(projn)
        Pn = P * An * prime(dag(An))
        if linkind_P === nothing
            prob = real(scalar(Pn))
        else
            prob = real(scalar(tracer * Pn))
        end
        pdisc += prob
        (random_number < pdisc) && break
        n += 1
    end
    return n, prob, Pn
end

function projective_measurement_gate_sample(s, result::Int, prob::Real; reset=nothing)
    sₚ = prime(s)
    projn = ITensor(s, sₚ)
    Zygote.ignore() do
        if reset === nothing
            projn[s => result, sₚ => result] = 1 / sqrt(prob)
        else
            @assert reset <= dim(s)
            projn[s => result, sₚ => reset] = 1 / sqrt(prob)
        end
    end
    return projn
end

function projective_measurement_sample(ψ::MPS; indices=1:length(ψ), reset::Union{Nothing, Integer, Vector{<:Integer}}=nothing, remove_measured=false, norm_treshold=0.9,
                                               get_projectors=false, get_loglike=false, gradient_averaging=true)
    #println("Warning: projective_measurement_sample needs to be validated")
    local N, result, P

    # In P we store the contracted left hand side of the tensor network
    # Diagramm:
    # P = O---O---O--
    #     |   |   |
    #     O---O---O--
    
    N = length(ψ)
    ψ = orthogonalize(ψ, 1)
    
    if reset === nothing || reset isa Int
        reset = fill(reset, length(indices))
    end

    n = norm(ψ[1])
    if abs(1 - n) < norm_treshold
        ψ[1] *= (1 / n)
    else
        error("sample: MPS is not normalized, norm=$(n), $(abs(1 - n)))> $norm_treshold")
    end

    result = zeros(Int, length(indices))
    projectors = Array{ITensor}(undef, length(indices))
    loglike = 0.

    i = 1
    for j in 1:N

        local prob, projn
        s = siteind(ψ, j)
        d = dim(s)

        if j in indices
            # Measure the qubit
            
            if j == 1
                # Diagramm P:
                #     O--
                #     |
                #     O
                #
                #     O
                #     |
                #     O--
                result[i], prob, An = sample_and_probs_mps(ψ[1], s, d)
                P = An * prime(dag(An))
                P *= (1 / prob)
            else
                # Diagramm P * ψj * ψj':
                #              O-- --O-- ψj
                #              O     |
                #              O     O proj
                #            P O      
                #              O     O proj
                #              O     |
                #              O-- --O-- ψj'

                linkind_P = linkind(ψ, j)
                result[i], prob, P = sample_and_probs_mps2(P, ψ[j], s, linkind_P, d)
                P *= (1 / prob)
            end
        

            # Get the projector
            projectors[i] = projective_measurement_gate_sample(s, result[i], prob; reset=reset[i])
            # Apply the projector
            ψ[j] = noprime(ψ[j] * projectors[i])
            loglike += log(prob)

            i += 1
        else
            # No measurement

            if j == 1
                # Diagramm P:
                #     O-- ψj
                #     |
                #     O-- ψj'
                linkind_P = linkind(ψ, j)
                P = ψ[1] * prime(dag(ψ[1]), linkind_P)
            elseif j < N
                # Diagramm P * ψj * ψj':
                #              O-- --O-- ψj
                #            P O     | 
                #              O-- --O-- ψj'
                linkind_P = linkind(ψ, j)
                linkind2_P = linkind(ψ, j-1)
                P = P * ψ[j] * prime(dag(ψ[j]), linkind_P, linkind2_P)
            end

        end

        # Sanity Check
        #=
        if j < N
            linkind_P = linkind(ψ, j)
            tracer = delta(linkind_P, prime(linkind_P))
            prob = real(scalar(tracer * P))
            @assert abs(1 - prob) < 1e-8
        end
        =#

    end
    println("a")
    println([eltype(ʋi) for ʋi in ψ])
    orthogonalize!(ψ, 1)
    println([eltype(ʋi) for ʋi in ψ])
    if remove_measured
        ψ = reduce_MPS(ψ, indices, result)
    end
    
    out = (ψ, result)
    if get_loglike
        out = (out..., loglike)
    end

    if get_projectors
        out = (out..., projectors)
    end
    return out
end


Zygote.@adjoint function projective_measurement_sample(ψ::MPS; indices=1:length(ψ), reset=nothing, remove_measured=false, 
                                                                        norm_treshold=0.9, get_projectors=false, get_loglike=false, gradient_averaging=true)
    
    """
    |ψₚ>=P(ψ)|ψ> where P is the measurment projector operator.
    If gradient_averaging=true the derivative is computed |ψₚ>=P|ψ> assuming P tobe constant, this is recomended if you want to average your gradients later as the average will be correct.
    If gradient_averaging=false the derivative is computed |ψₚ>=P(ψ)|ψ> assuming P to be a function of the parameters of the MPS, this is recomended if you want to use the gradient directly.
    P(ψ) = |r><M| / sqrt(<ψ|M><M|ψ>)
    exp(loglike/2) = sqrt(<ψ|M><M|ψ>)

    rrule gradient_averaging=true: <ψ_bar|∂|ψₚ>/∂|ψ> = <ψ_bar|P
    rrule gradient_averaging=false: <ψ_bar|∂|ψₚ>/∂|ψ> = <ψ_bar|P + <ψ_bar|∂|ψₚ>/∂P * ∂P/∂|ψ>
    <ψ_bar|∂|ψₚ>/∂P = |ψ_bar><ψ|
    ∂P/∂|ψ> = -|r><M| / <ψ|M><M|ψ>^(3/2) * |M><M|ψ>
    with Pₙ(ψ) = |M><M|
    <ψ_bar|∂|ψₚ>/∂P * ∂P/∂|ψ> = -Tr(|ψ_bar><ψ| * |M><r|) exp(-loglike*3/2) * Pₙ|ψ>
    <ψ_bar|∂|ψₚ>/∂P * ∂P/∂|ψ> = <ψ_bar|P|ψ> * exp(loglike) * exp(-loglike*3/2) * Pₙ|ψ>
    <ψ_bar|∂|ψₚ>/∂P * ∂P/∂|ψ> = <ψ_bar|P|ψ> * exp(-loglike/2) * Pₙ|ψ>
    """


    @assert remove_measured === false "Gradient with remove_measured=true not implemented"

    ψm, result, loglike, projectors = projective_measurement_sample(ψ; indices=indices, reset=reset, remove_measured=remove_measured,
                                                                    norm_treshold=norm_treshold, get_projectors=true, get_loglike=true)
    
    function f̄(ȳ)
        if length(ȳ) == 2
            ψ_bar, result_bar = ȳ
            loglike_bar = nothing
            @assert get_loglike === false
        else
            ψ_bar, result_bar, loglike_bar = ȳ
            @assert get_loglike === true
        end
        
        ψm_dag = conj(ψm)
        
        if !gradient_averaging # If gradients are not expected to be averaged and one want the true gradient
            if loglike_bar === nothing
                loglike_bar = 0.
            end
            loglike_bar -= inner(ψ_bar, ψm_dag) / 2.
        end

        @assert result_bar === nothing

        ψ_out = nothing
        projector_invs = [swapprime(projector, 0 => 1) for projector in projectors]

        if ψ_bar !== nothing
            ψ_out = copy(ψ_bar)
            for (i, index) in enumerate(indices)
                ψ_out[index] = noprime(ψ_out[index] * projector_invs[i])
            end
        end
        if loglike_bar != 0. && loglike_bar !== nothing
            f = 2 * exp(-loglike/2) * loglike_bar
            ψ_bar_prob = ψm_dag * f
            if reset !== nothing
                for (i, index) in enumerate(indices)
                    projector_inv_norm = projector_invs[i]  ./ norm(projector_invs[i])
                    ψ_bar_prob[index] = noprime(ψ_bar_prob[index] * projector_inv_norm)
                end
            end

            if ψ_out === nothing
                ψ_out = ψ_bar_prob
            else
                ψ_out += ψ_bar_prob
            end
            
        end
        return (ψ_out,)
    end

    out = (ψm, result)
    if get_loglike
        out = (out..., loglike)
    end
    if get_projectors
        out = (out..., projectors)
    end
    
    return out, f̄
end



"""
Projects the MPS ψ onto the state |n⟩, where n is a vector of integers. Eliminates the qubits that are measured.
"""
function reduce_MPS(ψ::MPS, indices::Vector{<:Integer}, values::Vector{<:Integer}; norm=false)
    N = length(ψ)
    ψ_tensors = ITensor[]
    j = 1
    P = nothing
    for i in 1:N
        if P === nothing
            ψi = ψ[i]
        else
            ψi = ψ[i] * P
            P = nothing
        end

        if i in indices
            s = siteind(ψ, i)
            projn = ITensor(s)
            projn[s => values[j]] = 1
            P = ψi * projn
            j += 1
        else
            ψ_tensors = vcat(ψ_tensors, ψi)
        end
    end
    if P !== nothing
        ψ_tensors = vcat(ψ_tensors[1:end-1], ψ_tensors[end] * P)
    end
    ψ = MPS(ψ_tensors)
    if norm
        ITensors.normalize!(ψ)
    end
    return ψ
end


function projective_measurement_sample_old(ψ::MPS; indices=1:length(ψ), reset=nothing, remove_measured=false, norm_treshold=0.9)
    # First sample the qubits and the apply the projectors seperately
    local N, result, P, projectors

    # In P we store the contracted left hand side of the tensor network
    # Diagramm:
    # P = O---O---O--
    #     |   |   |
    #     O---O---O--

    
    Zygote.@ignore begin 
        projectors = ITensor[]

        ψo = orthogonalize(ψ, 1)
        N = length(ψ)
        
        if ITensors.orthocenter(ψo) != 1
            error("sample: MPS ψ must have orthocenter(ψ)==1 and not $(ITensors.orthocenter(ψo))")
        end
        
        if reset === nothing || reset isa Int
            reset = fill(reset, length(indices))
        end
    
        #TODO: Check that the qubits are in the right order

        n = norm(ψo[1])
        if abs(1 - n) < norm_treshold
            ψo[1] *= (1 / n)
        else
            error("sample: MPS is not normalized, norm=$(n), $(abs(1 - n)))> $norm_treshold")
        end

        result = zeros(Int, length(indices))
    
        i = 1
        for j in 1:N

            local prob, projn
            s = siteind(ψo, j)
            d = dim(s)

            if j in indices
                # Measure the qubit
                Zygote.@ignore begin
                    if j == 1
                        # Diagramm P:
                        #     O--
                        #     |
                        #     O
                        #
                        #     O
                        #     |
                        #     O--
                        result[i], prob, An = sample_and_probs_mps(ψo[1], s, d)
                        P = An * prime(dag(An))
                        P *= (1 / prob)
                    else
                        # Diagramm P * ψj * ψj':
                        #              O-- --O-- ψj
                        #              O     |
                        #              O     O proj
                        #            P O      
                        #              O     O proj
                        #              O     |
                        #              O-- --O-- ψj'

                        linkind_P = linkind(ψo, j)
                        result[i], prob, P = sample_and_probs_mps2(P, ψo[j], s, linkind_P, d)
                        P *= (1 / prob)
                    end
                end

                # Get the projector
                projn = Zygote.@ignore projective_measurement_gate_sample(s, result[i], prob; reset=reset[i])
                push!(projectors, projn)
                # Apply the projector
                # = noprime(ψo[j] * projn)

                i += 1
            else
                # No measurement
                #ψj = ψ[j]

                Zygote.@ignore begin
                    if j == 1
                        # Diagramm P:
                        #     O-- ψj
                        #     |
                        #     O-- ψj'
                        linkind_P = linkind(ψo, j)
                        P = ψo[1] * prime(dag(ψo[1]), linkind_P)
                    elseif j < N
                        # Diagramm P * ψj * ψj':
                        #              O-- --O-- ψj
                        #            P O     | 
                        #              O-- --O-- ψj'
                        linkind_P = linkind(ψo, j)
                        linkind2_P = linkind(ψo, j-1)
                        P = P * ψo[j] * prime(dag(ψo[j]), linkind_P, linkind2_P)
                    end
                end
            end

            # Sanity Check
            #=
            if j < N
                linkind_P = linkind(ψ, j)
                tracer = delta(linkind_P, prime(linkind_P))
                prob = real(scalar(tracer * P))
                @assert abs(1 - prob) < 1e-8
            end
            =#

        end
    end

    ψ_new = apply(projectors, ψ)
    if remove_measured
        return reduce_MPS(ψ_new, indices, result), result
    else
        return ψ_new, result
    end
end