module ITensorsExtension
using ITensors
using ITensors: AbstractMPS

using PastaQ
using Zygote

# Types
VectorAbstractMPS = Union{Vector{MPS}, Vector{MPO}}
States = Union{VectorAbstractMPS, AbstractMPS} 

graderror(y; error_message="$y can not be differentiated") = y
Zygote.@adjoint function graderror(y; error_message="$y can not be differentiated")
    function pull(Δ)
        error(error_message)
        return (Δ)
    end
    return y, pull
end

function sample_and_probs_mps(A::ITensor, s, d; random_number=rand())
    local An, projn, pn

    pdisc = 0.
    n = 1
    while n <= d
        projn = ITensor(s)
        projn[s => n] = 1.
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

    pdisc = 0.
    n = 1
    # Remove the prime from the index A.tensor.inds[1]
    if linkind_P !== nothing
        tracer = delta(linkind_P, prime(linkind_P))
    end
    
    while n <= d
        projn = ITensor(s)
        projn[s => n] = 1.
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

function projective_measurement_sample(ψ::MPS; indices=1:length(ψ), reset=nothing, remove_measured=false, norm_treshold=0.9)
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
        if abs(1.0 - n) < norm_treshold
            ψo[1] *= (1.0 / n)
        else
            error("sample: MPS is not normalized, norm=$(n), $(abs(1.0 - n)))> $norm_treshold")
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
                        P *= (1. / prob)
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
                        P *= (1. / prob)
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
                @assert abs(1.0 - prob) < 1e-8
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

function projective_measurement_sample2(ψ::MPS; indices=1:length(ψ), reset=nothing, remove_measured=false, norm_treshold=0.9)
    # Orthocenter is at the end of the MPS so we need to sample from the end
    local result, P, random_numbers, norm_

    # In P we store the contracted left hand side of the tensor network
    # Diagramm:
    # P = O---O---O--
    #     |   |   |
    #     O---O---O--

    N = length(ψ)

    # Lazy way to orthogonalize the MPS
    ψ = runcircuit(ψ, [(("Id", N))])
    
    Zygote.@ignore begin 
        println("a4")
        if ITensors.orthocenter(ψ) != length(ψ)
            error("sample: MPS ψ must have orthocenter(ψ)==$(length(ψ)) and not $(ITensors.orthocenter(ψ))")
        end
        
        if reset === nothing || reset isa Int
            reset = fill(reset, length(indices))
        end
    
        #TODO: Check that the qubits are in the right order

        

        result = zeros(Int, length(indices))
        random_numbers = rand(length(indices))
    end

    norm_ = norm(ψ[end])
    if abs(1.0 - norm_) < norm_treshold
        # Normalize the MPS later
    else
        error("sample: MPS is not normalized, norm=$(norm_), $(abs(1.0 - norm_)))> $norm_treshold")
    end
    
    ψ_tensors = ITensor[]
    
    i = length(indices)
    for j in N:-1:1
        
        local prob, projn, ψj

        s = siteind(ψ, j)
        d = dim(s)
        ψj = ψ[j]
        if j == N
            ψj = ψj * (1/norm_)
        end

        if j in indices
            # Measure the qubit j
            
            Zygote.@ignore begin
                
                if j == N
                    # Diagramm P:
                    #     O--
                    #     |
                    #     O
                    #
                    #     O
                    #     |
                    #     O--
                    result[i], prob, An = sample_and_probs_mps(ψj, s, d; random_number=random_numbers[i])
                    P = An * prime(dag(An))
                    P *= (1. / prob)
                else
                    # Diagramm P * ψj * ψj':
                    #              O-- --O-- ψj
                    #              O     |
                    #              O     O proj
                    #            P O      
                    #              O     O proj
                    #              O     |
                    #              O-- --O-- ψj'

                    linkind_P = linkind(ψ, j-1)
                    result[i], prob, P = sample_and_probs_mps2(P, ψj, s, linkind_P, d; random_number=random_numbers[i])
                    P *= (1. / prob)
                end
            end
            

            # Get the projector
            projn = Zygote.@ignore projective_measurement_gate_sample(s, result[i], prob; reset=reset[i])
            
            # Apply the projector
            ψj = noprime(ψj * projn)

            i -= 1
        else
            # No measurement
            Zygote.@ignore begin
                if j == N
                    # Diagramm P:
                    #     O-- ψj
                    #     |
                    #     O-- ψj'
                    linkind_P = linkind(ψ, j-1)
                    P = ψj * prime(dag(ψj), linkind_P)
                elseif j > 1
                    # Diagramm P * ψj * ψj':
                    #              O-- --O-- ψj
                    #            P O     | 
                    #              O-- --O-- ψj'
                    linkind_P = linkind(ψ, j)
                    linkind2_P = linkind(ψ, j-1)
                    P = P * ψj * prime(dag(ψj), linkind_P, linkind2_P)
                end
            end
        end
        
        # Sanity Check
        #=
        if j < N
            linkind_P = linkind(ψ, j)
            tracer = delta(linkind_P, prime(linkind_P))
            prob = real(scalar(tracer * P))
            @assert abs(1.0 - prob) < 1e-8
        end
        =#
        ψ_tensors = vcat(ψ_tensors, ψj)

    end

    ψ_new = MPS(ψ_tensors[end:-1:1])

    if remove_measured
        return reduce_MPS(ψ_new, indices, result), result
    else
        return ψ_new, result
    end
end

function projective_measurement_sample!(ψ::MPS; indices=1:length(ψ), reset=nothing, remove_measured=false, norm_treshold=0.9)
    #println("Warning: projective_measurement_sample needs to be validated")
    local N, result, P

    # In P we store the contracted left hand side of the tensor network
    # Diagramm:
    # P = O---O---O--
    #     |   |   |
    #     O---O---O--

    #ψ = orthogonalize_grad(ψ, 1)
    Zygote.@ignore begin 
        N = length(ψ)
        orthogonalize!(ψ, 1)
        if ITensors.orthocenter(ψ) != 12
            error("sample: MPS ψ must have orthocenter(ψ)==1 and not $(ITensors.orthocenter(ψ))")
        end
        
        if reset === nothing || reset isa Int
            reset = fill(reset, length(indices))
        end
    
        #TODO: Check that the qubits are in the right order

        n = norm(ψ[1])
        if abs(1.0 - n) < norm_treshold
            ψ[1] *= (1.0 / n)
        else
            error("sample: MPS is not normalized, norm=$(n), $(abs(1.0 - n)))> $norm_treshold")
        end

        result = zeros(Int, length(indices))

    end

    ψ = graderror(ψ; error_message="projective_measurement_sample! can not be differentiated")
    
    ψ_tensors = ITensor[]
    
    i = 1
    for j in 1:N

        local prob, projn, ψj
        s = siteind(ψ, j)
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
                    result[i], prob, An = sample_and_probs_mps(ψ[1], s, d)
                    P = An * prime(dag(An))
                    P *= (1. / prob)
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
                    P *= (1. / prob)
                end
            end

            # Get the projector
            projn = Zygote.@ignore projective_measurement_gate_sample(s, result[i], prob; reset=reset[i])
            
            # Apply the projector
            ψj = noprime(ψ[j] * projn)

            i += 1
        else
            # No measurement
            ψj = ψ[j]

            Zygote.@ignore begin
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

        end

        # Sanity Check
        #=
        if j < N
            linkind_P = linkind(ψ, j)
            tracer = delta(linkind_P, prime(linkind_P))
            prob = real(scalar(tracer * P))
            @assert abs(1.0 - prob) < 1e-8
        end
        =#

        ψ_tensors = vcat(ψ_tensors, ψj)

    end

    ψ_new = MPS(ψ_tensors)
    if remove_measured
        return reduce_MPS(ψ_new, indices, result), result
    else
        return ψ_new, result
    end
end

"""
Projects the MPS ψ onto the state |n⟩, where n is a vector of integers. Eliminates the qubits that are measured.
"""
function reduce_MPS(ψ::MPS, indices::Vector{Int}, values::Vector{Int}; norm=false)
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
            projn[s => values[j]] = 1.
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
        projn[s => n] = 1.0
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
        if abs(1.0 - R[1][]) < norm_treshold
            R = R / norm_
            ρ[1] = ρ[1] / norm_
        else
            error("sample: MPO is not normalized, norm=$(tr(ρ)), $(abs(1.0 - R[1][]))> $norm_treshold")
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
        if abs(1.0 - R[1][]) < norm_treshold
            R = R / R[1][]
            ρ[1] = ρ[1] / R[1][]
        else
            error("sample: MPO is not normalized, norm=$(tr(ρ)), $(abs(1.0 - R[1][]))> $norm_treshold")
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


function projective_measurement_gate(s; reset=nothing)
    sₚ = prime(s)
    kraus = Index(s.space, "kraus")
    projn = ITensor(sₚ, s, kraus)
    Zygote.ignore() do
        if reset === nothing   
            for l in 1:s.space
                projn[sₚ => l, s => l, kraus => l] = 1.
            end
        else
            for l in 1:s.space
                projn[sₚ => reset, s => l, kraus => l] = 1.
            end
        end
    end
    return projn
end


function projective_measurement_gate_sample(s, result::Int, prob::Real; reset=nothing)
    sₚ = prime(s)
    projn = ITensor(s, sₚ)
    Zygote.ignore() do
        if reset === nothing
            projn[s => result, sₚ => result] = 1. / sqrt(prob)
        else
            @assert reset <= dim(s)
            projn[s => result, sₚ => reset] = 1. / sqrt(prob)
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

function add_identities(H::MPO, hilbert, sites)
    @assert length(hilbert) == length(sites)
    H = H[:]
    
    for (index, site) in zip(hilbert, sites)
        local new_H
        
        if site == 1 || site > length(H)
            new_H = δ(index, index')
        else
            link_original = commonind(H[site-1], H[site])
            link_new = Index(dim(link_original); tags="Link,n=e$site")
            H[site] = H[site] * δ(link_original, link_new)

            new_H = δ(link_original, link_new) * δ(index, index')
        end
            
            
        insert!(H, site, new_H)
    end
    return MPO(H)
    
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
                        projn[s1 => l, s2 => l, s1ₚ => l, s2ₚ => l] = 1.
                    end
                else
                    for l in 1:s1.space
                        projn[s1 => l, s2 => l, s1ₚ => reset[j], s2ₚ => reset[j]] = 1.
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

using ITensors: AbstractMPS
function orthogonalize_grad(M::AbstractMPS, j::Int; kwargs...)
    llim = M.llim
    rlim = M.rlim
    M_O = M
    M = Zygote.bufferfrom(M[:])
    
    if !(1 <= j <= length(M))
      error("Input j=$j to `orthogonalize!` out of range (valid range = 1:$(length(M)))")
    end
    
    
    while llim < (j - 1)
        if llim < 0
            llim = 0
        end
        b = llim + 1
        linds = uniqueinds(M[b], M[b + 1])
        lb = linkind(M_O, b)
        if !isnothing(lb)
          ltags = tags(lb)
        else
            ltags = TagSet("Link,l=$b")
        end
        L, R = ITensors.factorize(M[b], linds; tags=ltags, kwargs...)
        M[b] = L
        M[b + 1] *= R
        llim = b
        if rlim < llim + 2
           rlim = llim + 2
        end
    end

    N = length(M)

    while rlim > (j + 1)
        if rlim > (N + 1)
            rlim = N + 1
        end
        b = rlim - 2
        rinds = uniqueinds(M[b + 1], M[b])
        lb = linkind(M_O, b)
        if !isnothing(lb)
            ltags = tags(lb)
        else
            ltags = TagSet("Link,l=$b")
        end
        L, R = factorize(M[b + 1], rinds; tags=ltags, kwargs...)
        M[b + 1] = L
        M[b] *= R

        rlim = b + 1
        if llim > rlim - 2
            llim = rlim - 2
        end
    end
    return MPS(copy(M), llim, rlim)
end


# Custom @adoints for Zygote
function Base.convert(::Type{T}, x::Zygote.Tangent) where {T <: ITensors.AbstractMPS}
    return T(x.data, x.llim, x.rlim)
end

function Base.convert(::Type{Vector{ITensor}}, x::States)
    return x.data
end

@Zygote.adjoint ITensors.MPO(data::Vector{ITensor}, llim::Int, rlim::Int) = MPO(data, llim, rlim), c̄ -> (MPO(c̄.data, Zygote.ChainRulesCore.ZeroTangent(), Zygote.ChainRulesCore.ZeroTangent()),)
@Zygote.adjoint ITensors.MPO(data::Vector{ITensor}; ortho_lims::UnitRange=1:length(data)) = MPO(data; ortho_lims), c̄ -> (MPO(c̄.data, Zygote.ChainRulesCore.ZeroTangent(), Zygote.ChainRulesCore.ZeroTangent()),)

@Zygote.adjoint ITensors.MPS(data::Vector{ITensor}, llim::Int, rlim::Int) = MPS(data, llim, rlim), c̄ -> (MPS(c̄.data, Zygote.ChainRulesCore.ZeroTangent(), Zygote.ChainRulesCore.ZeroTangent()),)
@Zygote.adjoint ITensors.MPS(data::Vector{ITensor}; ortho_lims::UnitRange=1:length(data)) = MPS(data; ortho_lims), c̄ -> (MPS(c̄.data, Zygote.ChainRulesCore.ZeroTangent(), Zygote.ChainRulesCore.ZeroTangent()),)



# Make printing easier
function Base.println(inds::NTuple{T, Index{Int64}} where {T})
    for (i, s) in enumerate(inds)
        println("$i $s")
    end
end

# end module
end # module