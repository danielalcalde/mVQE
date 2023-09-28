struct nMPS <: AbstractMPS
    ψ::AbstractMPS
end

function Base.getproperty(ψ::nMPS, f::Symbol)
    if f == :ψ
        return getfield(ψ, f)
    else
        return getfield(ψ.ψ, f)
    end
end

expect(ψ::MPS, ψ2::nMPS; kwargs...) = 1 .- expect(ψ, ψ2.ψ; kwargs...)

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