struct nMPS <: AbstractMPS
    ψ::AbstractMPS
    mode::Symbol
    nMPS(ψ::AbstractMPS; mode::Symbol=:fid) = new(ψ, mode)
end

function Base.getproperty(ψ::nMPS, f::Symbol)
    if f == :ψ || f == :mode
        return getfield(ψ, f)
    else
        return getfield(ψ.ψ, f)
    end
end

function expect(ψ::MPS, ψ2::nMPS; kwargs...)
    f = expect(ψ, ψ2.ψ; kwargs...)
    if ψ2.mode == :fid
        return 1. - f
    elseif ψ2.mode == :logfid
        return log(1. - f) - log(f)
    else
        error("Mode $(ψ2.mode) not recognized for nMPS. Only :fid and :logfid are supported.")
    end
end

struct BasisnMPS <: AbstractMPS
    ψs::Vector{<:AbstractMPS}
end

function Base.getproperty(ψ::BasisnMPS, f::Symbol)
    if f == :ψs
        return getfield(ψ, f)
    else
        ψs = getfield(ψ, :ψs)
        return getfield(ψs[1], f)
    end
end

expect(ψ::MPS, ψ2::BasisnMPS; kwargs...) = 1 .- sum([expect(ψ, ψi; kwargs...) for ψi in ψ2.ψs])