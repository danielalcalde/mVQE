module ITensorsExtension
using ITensors
using ITensors: AbstractMPS

using PastaQ
using Zygote

# Types
VectorAbstractMPS = Union{Vector{MPS}, Vector{MPO}}
States = Union{VectorAbstractMPS, AbstractMPS} 



function projective_measurement_sample(ψ::MPS; indices=1:length(ψ), reset=nothing)
    N = length(ψ)
    orthogonalize!(ψ, 1)
    if ITensors.orthocenter(ψ) != 1
        error("sample: MPS ψ must have orthocenter(ψ)==1")
    end
    
    if reset === nothing || reset isa Int
        reset = fill(reset, length(indices))
    end
    
    #TODO: Check that the qubits are in order
    
    if abs(1.0 - norm(ψ[1])) > 1E-8
        error("sample: MPS is not normalized, norm=$(norm(ψ[1]))")
    end
    
    result = zeros(Int, length(indices))
    A = ψ[1]
    
    ψ_tensors = ITensor[]
    
    i = 1
    for j in 1:N
        s = siteind(ψ, j)
        d = dim(s)
        # Compute the probability of each state
        # one-by-one and stop when the random
        # number r is below the total prob so far
        
        # Will need An bellow
        An = ITensor()
        
        if j in indices
            pdisc = 0.0
            r = rand()
            pn = 0.0
            n = 1
            while n <= d
                projn = ITensor(s)
                projn[s => n] = 1.0
                An = A * dag(projn)
                pn = real(scalar(dag(An) * An))
                pdisc += pn
                (r < pdisc) && break
                n += 1
            end
            
            # Set the new mps
            projn = projective_measurement_gate_mc(s, n, pn; reset=reset[i])
            
            ψj = noprime(ψ[j] * projn)
            
            result[i] = n
            i += 1
        else
            # No measurement
            ψj = ψ[j]
            An = A
            pn = 1.
        end

        ψ_tensors = vcat(ψ_tensors, ψj)
        
        if j < N
            A = ψ[j + 1] * An
            A *= (1.0 / sqrt(pn))
        end
    end
    return MPS(ψ_tensors), result
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

function Base.convert(ty::Type{Vector{ITensor}}, x::MPO)
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