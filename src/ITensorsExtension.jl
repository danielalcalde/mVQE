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