module ITensorsExtension
using ITensors
using ITensors: AbstractMPS

using PastaQ
using Zygote

# Types
VectorAbstractMPS = Union{Vector{MPS}, Vector{MPO}}
States = Union{VectorAbstractMPS, AbstractMPS} 


function add_identities(H::MPO, hilbert, sites)
    @assert length(hilbert) == length(sites)
    H = H[:]
    
    for (index, site) in zip(hilbert, sites)
        new_H = δ(index, index')
        
        if !(site == 1 || site > length(H))
            link_original = commonind(H[site-1], H[site])
            if link_original !== nothing
                link_new = Index(dim(link_original); tags="Link,n=e$site")
                H[site] = H[site] * δ(link_original, link_new)

                new_H = δ(link_original, link_new) * new_H
            else
                @assert length(inds(H[site-1])) == 2
            end
        end

        insert!(H, site, new_H)
    end
    return MPO(H)
end

function get_projector(hilbert, ancilla_indices, bit_array::Vector{<:Integer})
    return Zygote.@ignore begin 
        state_indices = setdiff(1:length(hilbert), ancilla_indices)
        ψp_a = productstate(hilbert[ancilla_indices], bit_array)
        Oψp_a = outer(ψp_a, ψp_a')
        return add_identities(Oψp_a, hilbert[state_indices], state_indices)
    end
end

function get_mapper(hilbert, ancilla_indices, input_bit_array::Vector{<:Integer}, output_bit_array::Vector{<:Integer})
    return Zygote.@ignore begin 
        state_indices = setdiff(1:length(hilbert), ancilla_indices)
        ψp_a = productstate(hilbert[ancilla_indices], input_bit_array)
        ψp_a2 = productstate(hilbert[ancilla_indices], output_bit_array)
        Oψp_a = outer(ψp_a, ψp_a2')
        return add_identities(Oψp_a, hilbert[state_indices], state_indices)
    end
end

function add_ancillas(ψ::MPS, hilbert, sites; state=1)
    @assert length(hilbert) == length(sites)
    ψ = ψ[:]
    if state isa Int
        state = fill(state, length(sites))
    end
    
    for (state_i, index, site) in zip(state, hilbert, sites)
        new_ψ = ITensor(index)
        new_ψ[index => state_i] = 1.

        if !(site == 1 || site > length(ψ))
            link_original = commonind(ψ[site-1], ψ[site])
            link_new = Index(dim(link_original); tags="Link,n=e$site")
            ψ[site] = ψ[site] * δ(link_original, link_new)

            new_ψ = δ(link_original, link_new) * new_ψ
        end
        
        insert!(ψ, site, new_ψ)
    end
    return MPS(ψ)
end

function add_ancillas(mpo::MPO, hilbert, sites)
    @assert length(hilbert) == length(sites)
    mpo = mpo[:]
    
    for (index, site) in zip(hilbert, sites)
        new_mpo = ITensor(index)
        new_mpo = δ(index, index')

        if !(site == 1 || site > length(mpo))
            link_original = commonind(mpo[site-1], mpo[site])
            link_new = Index(dim(link_original); tags="Link,n=e$site")
            mpo[site] = mpo[site] * δ(link_original, link_new)

            new_mpo = δ(link_original, link_new) * new_mpo
        end
        
        insert!(mpo, site, new_mpo)
    end
    return MPO(mpo)
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

# Make printing easier
function Base.println(inds::NTuple{T, Index{Int64}} where {T})
    for (i, s) in enumerate(inds)
        println("$i $s")
    end
end

# end module
end # module