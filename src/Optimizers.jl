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
    losstol::Real
    verbosity::Int
    OptimizerWrapper(optimizer; maxiter=50, losstol=-10^10, gradtol=1e-10, verbosity=0) = new(optimizer, maxiter, gradtol, losstol, verbosity)
end

callback_(args...; kwargs...) = nothing

function OptimKit.optimize(loss_and_grad, θ::T, optimizer::OptimizerWrapper,
                           callback = (args...; kwargs...) -> nothing;
                          copy=true,
                          finalize! = OptimKit._finalize!, precondition=OptimKit._precondition
                          ) where {T}
    if copy
        θ = deepcopy(θ)
    end
    
    history = Matrix{Float64}(undef, optimizer.maxiter, 3)

    local loss, gradient_, niter_
    for niter in 1:optimizer.maxiter
        niter_ = niter
        loss, gradient_ = loss_and_grad(θ)
        norm_grad = norm(gradient_)

        # Precondition
        gradient_ = precondition(θ, deepcopy(gradient_))

        # Finalize
        θ, loss, gradient_ = finalize!(θ, loss, gradient_, niter)
        
        # check if isa vector
        if hasmethod(Flux.params, (T,)) && ! (θ isa Vector)
            # It is a flux model
            Flux.update!(optimizer.optimizer, Flux.params(θ), gradient_)
            norm_θ = norm(Flux.params(θ))
        else
            Flux.update!(optimizer.optimizer, θ, gradient_)
            norm_θ = norm(θ)
        end

        # Saving the loss and norms
        history[niter, :] .= loss, norm_grad, norm_θ

        # Callback
        misc = Dict("loss" => loss, "gradient" => gradient_, "niter" => niter, "history" => history[1:niter, :])
        callback(; loss_value=loss, model=θ, misc=misc, niter=niter)
        
        if optimizer.verbosity >= 2
            @info "$(typeof(optimizer.optimizer)): iter $niter: f = $loss, ‖∇f‖ = $(norm_grad), ‖θ‖ = $(norm_θ)"
            flush(stdout)
            flush(stderr)
        end

        if norm_grad < optimizer.gradtol
            @info "Gradient tolerance reached"
            break
        end

        if loss < optimizer.losstol
            @info "Loss tolerance reached"
            break
        end
    end

    if optimizer.verbosity == 1
        @info "$(typeof(optimizer.optimizer)): iter $niter: f = $loss, ‖∇f‖ = $(norm_grad), ‖θ‖ = $(norm_θ)"
        flush(stdout)
        flush(stderr)
    end

    # Truncating the history
    history = history[1:niter_, :]

    return θ, loss, gradient_, niter_, history
end



# Change the OptimKit.optimize function to work with callbacks
function OptimKit.optimize(loss_and_grad, θ::T, optimizer::OptimKit.OptimizationAlgorithm,
                           callback;
                           copy=true, finalize! = OptimKit._finalize!, kwargs... 
                           ) where {T}
    if copy
        θ = deepcopy(θ)
    end
    
    history = Matrix{Float64}(undef, optimizer.maxiter, 3)

    function finalize_2!(θ, loss, gradient_, niter)
        norm_grad = norm(gradient_)
        local norm_θ
        if hasmethod(Flux.params, (T,))
            norm_θ = norm(Flux.params(θ))
        else
            norm_θ = norm(θ)
        end

        θ, loss, gradient_ = finalize!(θ, loss, gradient_, niter)
        history[niter, :] .= loss, norm_grad, norm_θ

        # Callback
        misc = Dict("loss" => loss, "gradient" => gradient_, "niter" => niter, "history" => history[1:niter, :])
        callback(; loss_value=loss, model=θ, misc=misc, niter=niter)
        return θ, loss, gradient_
    end
    θ, loss, gradient_, niter_, history2 = OptimKit.optimize(loss_and_grad, θ, optimizer;
                                                             finalize! = finalize_2!, kwargs...)
    
    return θ, loss, gradient_, niter_, history
end


# Functions to make a Flux model work with OptimKit
import Base.*
import Base./
import Base.+
import Base.-


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
        if gs[vi] !== nothing
            gs[vi] = gs[vi] ./ α
        end
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

#=
function +(grads1::Zygote.Grads, grads2::Zygote.Grads)
    @assert grads1.params == grads2.params "The gradients do not have the same parameters"
    out = copy(grads1)º
    for p in grads1.params
        out.grads[p] = grads1[p] .+ grads2[p]
    end
    return out
end
=#

function +(grads1::Zygote.Grads, grads2::Zygote.Grads) # This is the one that is used, fix so that it works with Distributed
    @assert grads1.params == grads2.params "The gradients do not have the same parameters"
    out = copy(grads1)
    for i in 1:length(out.params)
        @assert out.params[i] == grads1.params[i] == grads2.params[i] "The gradients do not have the same parameters"
        out.grads[out.params[i]] = grads1[grads1.params[i]] .+ grads2[grads2.params[i]]
    end
    return out
end

function -(grads1::Zygote.Grads, grads2::Zygote.Grads) # This is the one that is used, fix so that it works with Distributed
    @assert grads1.params == grads2.params "The gradients do not have the same parameters"
    out = copy(grads1)
    for i in 1:length(out.params)
        @assert out.params[i] == grads1.params[i] == grads2.params[i] "The gradients do not have the same parameters"
        out.grads[out.params[i]] = grads1[grads1.params[i]] .- grads2[grads2.params[i]]
    end
    return out
end

function *(grads1::Zygote.Grads, grads2::Zygote.Grads) # This is the one that is used, fix so that it works with Distributed
    @assert grads1.params == grads2.params "The gradients do not have the same parameters"
    out = copy(grads1)
    for i in 1:length(out.params)
        @assert out.params[i] == grads1.params[i] == grads2.params[i] "The gradients do not have the same parameters"
        out.grads[out.params[i]] = grads1[grads1.params[i]] .* grads2[grads2.params[i]]
    end
    return out
end

function Base.sqrt(grads::Zygote.Grads)
    grads2 = copy(grads)
    for p in grads.params
        grads2[p] = sqrt.(deepcopy(grads[p]))
    end
    return grads2
end

function Base.deepcopy(grads::Zygote.Grads)
    grads = copy(grads)
    for p in grads.params
        grads[p] = deepcopy(grads[p])
    end
    return grads
end



end # module