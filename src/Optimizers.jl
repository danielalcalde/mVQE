module Optimizers

using Flux: update!
using Flux.Optimise: AbstractOptimiser
using OptimKit
using LinearAlgebra

struct OptimizerWrapper
    optimizer::AbstractOptimiser
    maxiter::Int
    gradtol::Real
    verbosity::Int
    OptimizerWrapper(optimizer; maxiter=50, gradtol=1e-10, verbosity=0) = new(optimizer, maxiter, gradtol, verbosity)
end


function OptimKit.optimize(loss_and_grad, θ, optimizer::OptimizerWrapper; finalize! = OptimKit._finalize!)
    θ = copy(θ)
    
    history = Matrix{Float64}(undef, optimizer.maxiter, 2)

    niter = 0
    local loss, gradient_
    for niter in 1:optimizer.maxiter
        loss, gradient_ = loss_and_grad(θ)
        update!(optimizer.optimizer, θ, gradient_)
        
        normgrad = norm(gradient_)
        if normgrad < optimizer.gradtol
            break
        end

        # Saving the loss and gradient
        history[niter, :] .= loss, normgrad
        
        if optimizer.verbosity >= 2
            @info "$(typeof(optimizer)): iter $niter f = $loss ‖∇f‖ = $(normgrad)"
            flush(stdout)
            flush(stderr)
        end

        θ, loss, gradient_ = finalize!(θ, loss, gradient_, niter)
    end

    if optimizer.verbosity == 1
        @info "$(typeof(optimizer)): iter $niter f = $loss ‖∇f‖ = $(normgrad)"
        flush(stdout)
        flush(stderr)
    end
    
    return θ, loss, gradient_, niter, history
end

import Base.*
import Base./
import Base.+

function Base.zero(x::Vector{T}) where{T}
    return [zero(xi) for xi in x]
end

function Base.zero(::Type{Vector{Float64}}) where{T}
    return [zero(xi) for xi in x]
end

function *(x1::Vector{T}, x2::Vector{T}) where{T}
    return [x1i .* x2i for (x1i, x2i) in zip(x1, x2)]
end

function /(x1::Vector{T}, x2::Vector{T}) where{T}
    return [x1i ./ x2i for (x1i, x2i) in zip(x1, x2)]
end

function +(x1::Vector{T}, x2::Vector{T}) where{T}
    return [x1i .+ x2i for (x1i, x2i) in zip(x1, x2)]
end

function +(x1::Vector{T}, x2::Real) where{T}
    return [xi .+ x2 for xi in x1]
end

function Base.sqrt(x::Vector{T}) where{T}
    return [sqrt.(xi) for xi in x]
end

end # module