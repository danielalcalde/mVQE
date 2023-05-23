module MPOExtensions
using ITensors
using ITensors: terms, sortmergeterms, which_op, site, params
using ITensors: AbstractMPS

# Fake until implement
struct PartialMPO <: AbstractMPS
    mpo::MPO
    sites_lookup::Dict
    function PartialMPO(op::OpSum, hilbert)
        sites_ = sites(op)
        sites_lookup = Dict(s=>i for (i, s) in enumerate(sites_))
            
        return new(MPO(op, hilbert), sites_lookup)
    end
end

# General
ITensors.expect(ρ::MPO, mpo::MPO; kwargs...) = inner(ρ, mpo; kwargs...)
ITensors.expect(ψ::MPS, mpo::MPO; kwargs...) = inner(ψ', mpo, ψ; kwargs...)

# Specific
ITensors.expect(ψ::MPS, O::PartialMPO; kwargs...) = expect(ψ, O.mpo; kwargs...)
ITensors.expect(ψ::MPS, O::Vector{PartialMPO}; kwargs...) = sum(expect(ψ, Oi; kwargs...) for Oi in O) 

ITensors.expect(ψ::MPO, O::PartialMPO; kwargs...) = expect(ψ, O.mpo; kwargs...)
ITensors.expect(ψ::MPO, O::Vector{PartialMPO}; kwargs...) = sum(expect(ψ, Oi; kwargs...) for Oi in O) 
ITensors.data(O::PartialMPO) = ITensors.data(O.mpo)

struct PartialMPODummy <: AbstractMPS
    mpo::MPO
    sites_lookup::Dict
    function PartialMPODummy(op::OpSum, hilbert::Vector{Index{T}} where T<:Integer)
        sites_ = sites(op)
        op = site_map(op, sites_)
        
        sites_lookup = Dict(s=>i for (i, s) in enumerate(sites_))
            
        return new(MPO(op, hilbert[sites_]), sites_lookup)
    end
end
sites(x::PartialMPODummy) = sort(collect(keys(x.sites_lookup)))
function Base.getindex(x::PartialMPODummy, i::Int)
    @assert i in keys(x.sites_lookup) "Site $i was not found in PartialMPODummy it has sites $(sites(x))"
    return x.mpo[x.sites_lookup[i]]
end

function sites(op::OpSum)
    sites_ = []
    for opi in op
        for opij in opi.args[2]
            s = site(opij)
            if ! (s in sites_)
                push!(sites_, s)
            end
        end
    end
    sort!(sites_)
    return sites_
end

function site_map(op1::OpSum, sites_lookup)
    if sites_lookup isa Vector
        sites_lookup = Dict(s=>i for (i, s) in enumerate(sites_lookup))
    end
    
    op2 = OpSum()
    for opi in op1
        p = Prod{Op}()
        for opij in opi.args[2]
            p *= Op(which_op(opij), sites_lookup[site(opij)])
        end
        args = (opi.args[1], p)
        opi_mod = Scaled{ComplexF64, Prod{Op}}(opi.f, args, opi.kwargs)
        op2 += opi_mod
    end
    return op2
end

function ITensors.expect(ψ::MPS, O::PartialMPODummy)
    O_sites = sites(O)
    
    @assert O_sites[1] <= ψ.llim+1 &&  ψ.rlim-1 <= O_sites[end] "The orthogonality center needs to be between $(O_sites[1])-$(O_sites[end]) but is between $(ψ.llim)-$(ψ.rlim)"
    
    ψd = dag(ψ)
    ψd = ψd'
    link = linkind(ψ, O_sites[1]-1)
    
    r = ψd[O_sites[1]] * δ(link', link) * O[O_sites[1]] * ψ[O_sites[1]]
    
    for i in O_sites[1]+1:O_sites[end]-1
        if i in O_sites
            r = r * ψd[i] * O[i] * ψ[i]
        else
            s = siteind(ψ, i)
            r = r * ψd[i] * δ(s', s) * ψ[i]
        end
    end
    
    link = linkind(ψ, O_sites[end])
    r = r * ψd[O_sites[end]] * δ(link', link) * O[O_sites[end]] * ψ[O_sites[end]] 
    
    return r[]
end

function expect_and_orthogonalize(ψ::MPS, O::PartialMPODummy)
    O_sites = sites(O)
    
    if ! (O_sites[1] <= ψ.llim+1)
        ψ = orthogonalize(ψ, O_sites[1])
    end
    if ! (ψ.rlim-1 <= O_sites[end])
        ψ = orthogonalize(ψ, O_sites[end])
    end
    
    return expect(ψ, O), ψ
    
end



end # module