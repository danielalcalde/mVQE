module ITensorsExtension
using ITensors
using ITensors: AbstractMPS

using PastaQ
using Zygote

# Types
VectorAbstractMPS = Union{Vector{MPS}, Vector{MPO}}
States = Union{VectorAbstractMPS, AbstractMPS} 


function sample_and_probs_mps(A::ITensor, s, d)
    local An, projn, pn

    pdisc = 0.
    r = rand()
    n = 1
    while n <= d
        projn = ITensor(s)
        projn[s => n] = 1.
        An = A * dag(projn)
        pn = real(scalar(dag(An) * An))
        pdisc += pn
        (r < pdisc) && break
        n += 1
    end
    return n, pn, An
end

function sample_and_probs_mps2(P::ITensor, ψi::ITensor, s, linkind_P, d)
    local An, projn, prob, Pn

    pdisc = 0.
    r = rand()
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
        (r < pdisc) && break
        n += 1
    end
    return n, prob, Pn
end

function projective_measurement_sample(ψ::MPS; indices=1:length(ψ), reset=nothing, remove_measured=false, norm_treshold=0.9)
    #println("Warning: projective_measurement_sample needs to be validated")
    local N, result, P

    # In P we store the contracted left hand side of the tensor network
    # Diagramm:
    # P = O---O---O--
    #     |   |   |
    #     O---O---O--

    Zygote.@ignore begin 
        N = length(ψ)

        orthogonalize!(ψ, 1)
        if ITensors.orthocenter(ψ) != 1
            error("sample: MPS ψ must have orthocenter(ψ)==1")
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
        if imag(pnc) > 1e-8
            @warn "In sample, probability $pnc is complex."
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
        if abs(1.0 - R[1][]) < norm_treshold
            R = R / R[1][]
            ρ[1] = ρ[1] / R[1][]
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

function projective_measurement_sample2(ρ::MPO; indices=1:length(ρ), reset=nothing)
    gates = Vector{ITensor}(undef, length(indices))
    result = Vector{Int}(undef, length(indices))

    Zygote.ignore() do
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

    return MPO(ρ_tensors), nothing
end


function projective_measurement(ψ::MPS; kwargs...)
    ρ = outer(ψ, ψ')
    ρ, samples = projective_measurement(ρ; kwargs...)
    return ρ, samples
end


function projective_measurement(ψs::Vector{MPS}; kwargs...) 
    ψs_out = MPS[]
    res_out = Vector{Vector{Int64}}()
    for ψ in ψs
        ψ, res = projective_measurement(ψ; kwargs...)
        ψs_out = vcat(ψs_out, [ψ])
        res_out = vcat(res_out, [res])
    end

    return ψs_out, res_out
end

function projective_measurement(ρs::Vector{MPO}; kwargs...)
    ρs_out = MPO[]
    for ρ in ρs
        ρ = projective_measurement(ρ; kwargs...)[1]
        ρs_out = vcat(ρs_out, [ρ])
    end

    return ρs_out, nothing
end


# Custom @adoints for Zygote
function Base.convert(ty::Type{MPO}, x::Zygote.Tangent)
    return MPO(x.data, x.llim, x.rlim)
end

function Base.convert(ty::Type{MPS}, x::Zygote.Tangent)
    return MPS(x.data, x.llim, x.rlim)
end

function Base.convert(ty::Type{Vector{ITensor}}, x::States)
    return x.data
end

@Zygote.adjoint MPO(data::Vector{ITensor}, llim::Int, rlim::Int) = MPO(data, llim, rlim), c̄ -> (MPO(c̄.data, Zygote.ChainRulesCore.ZeroTangent(), Zygote.ChainRulesCore.ZeroTangent()),)
@Zygote.adjoint MPO(data::Vector{ITensor}; ortho_lims::UnitRange=1:length(data)) = MPO(data; ortho_lims), c̄ -> (MPO(c̄.data, Zygote.ChainRulesCore.ZeroTangent(), Zygote.ChainRulesCore.ZeroTangent()),)



# Make printing easier
function Base.println(inds::NTuple{T, Index{Int64}} where {T})
    for (i, s) in enumerate(inds)
        println("$i $s")
    end
end

# end module
end # module