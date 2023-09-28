expect(ψ::MPS, H::MPO; kwargs...) = real(inner(ψ', H, ψ; kwargs...))
expect(ρ::MPO, H::MPO; kwargs...) = real(inner(ρ, H; kwargs...))
expect(ψ::MPS, ψ2::MPS; kwargs...) = abs2(inner(ψ, ψ2; kwargs...))


Zygote.@adjoint function expect(ψ::MPS, H::MPO; kwargs...)
    function f̄(ȳ)
        ψbar = apply(H, ψ; kwargs...)
        return (2 * ȳ) * ψbar, nothing
    end
   return expect(ψ, H; kwargs...), f̄
end