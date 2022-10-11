module Optimizers

import Flux
using Flux.Optimise: AbstractOptimiser
using OptimKit
using LinearAlgebra
import Zygote

struct OptimizerWrapper
    optimizer::AbstractOptimiser
    maxiter::Int
    gradtol::Real
    verbosity::Int
    OptimizerWrapper(optimizer; maxiter=50, gradtol=1e-10, verbosity=0) = new(optimizer, maxiter, gradtol, verbosity)
end

function OptimKit.optimize(loss_and_grad, θ::T, optimizer::OptimizerWrapper; finalize! = OptimKit._finalize!) where {T}
    θ = deepcopy(θ)
    
    history = Matrix{Float64}(undef, optimizer.maxiter, 2)

    niter = 0
    local loss, gradient_
    for niter in 1:optimizer.maxiter
        loss, gradient_ = loss_and_grad(θ)
        normgrad = norm(gradient_)
        
        if hasmethod(Flux.params, (T,))
            # It is a flux model
            Flux.update!(optimizer.optimizer, Flux.params(θ), gradient_)
        else
            Flux.update!(optimizer.optimizer, θ, gradient_)
        end
        
        if normgrad < optimizer.gradtol
            break
        end

        # Saving the loss and gradient
        history[niter, :] .= loss, normgrad
        
        if optimizer.verbosity >= 2
            @info "$(typeof(optimizer.optimizer)): iter $niter: f = $loss, ‖∇f‖ = $(normgrad)"
            flush(stdout)
            flush(stderr)
        end

        θ, loss, gradient_ = finalize!(θ, loss, gradient_, niter)
    end

    if optimizer.verbosity == 1
        @info "$(typeof(optimizer.optimizer)): iter $niter: f = $loss, ‖∇f‖ = $(normgrad)"
        flush(stdout)
        flush(stderr)
    end
    
    return θ, loss, gradient_, niter, history
end


# Functions to make 
import Base.*
import Base./
import Base.+

function LinearAlgebra.rmul!(gs::Zygote.Grads, α::Number)
    for vi in gs.params
        LinearAlgebra.rmul!(gs[vi], α)
    end
    return gs
end

function *(gs::Zygote.Grads, α::Number)
    gs = copy(gs)
    for vi in gs.params
        gs[vi] = gs[vi] .* α
    end
    return gs
end

function /(gs::Zygote.Grads, α::Number)
    gs = copy(gs)
    for vi in gs.params
        gs[vi] = gs[vi] ./ α
    end
    return gs
end

*(α::Number, gs::Zygote.Grads) = gs * α

function LinearAlgebra.axpy!(α::Number, vsrc::Zygote.Grads, vdst::Zygote.Grads)
    #vdst += vsrc * α
    for i in 1:length(vdst.params)
        vdst.grads[vdst.params[i]] .+= vsrc.grads[vsrc.params[i]] .* α
    end
    return vdst
end

function +(model::T, grads::Zygote.Grads) where {T}
    @assert hasmethod(Flux.params, (T,)) "The model does not have a params method"
    model_new = deepcopy(model)
    params = Flux.params(model_new)
    
    for i in 1:length(params)
        params[i] .+= grads[grads.params[i]]
    end
    return model_new
end

function +(grads1::Zygote.Grads, grads2::Zygote.Grads)
    @assert grads1.params == grads2.params "The gradients do not have the same parameters"

    out = copy(grads1)
    for p in grads1.params
        out.grads[p] = grads1[p] .+ grads2[p]
    end
    return out
end

function Base.deepcopy(grads::Zygote.Grads)
    grads = copy(grads)
    for p in grads.params
        grads[p] = deepcopy(grads[p])
    end
    return grads
end

end # module