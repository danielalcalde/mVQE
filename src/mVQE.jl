module mVQE

using ThreadsX
using Distributed
using ParallelDataTransfer

using PastaQ
using ITensors
using Random
using OptimKit
using Zygote
using Statistics
import Flux

using ITensors: AbstractMPS

include("Optimizers.jl")
include("Misc.jl")
include("Hamiltonians.jl")

include("ITensorsExtension.jl")

include("StateFactory.jl")

include("Gates.jl")
include("Layers.jl")
include("Circuits.jl")

using mVQE.ITensorsExtension: VectorAbstractMPS, States, projective_measurement
using mVQE.Circuits: AbstractVariationalCircuit, AbstractVariationalMeasurementCircuit


# Cost function
function loss(ψ::MPS, H::MPO; kwargs...)
    return real(inner(ψ', H, ψ; kwargs...))
end

function loss(ρ::MPO, H::MPO; kwargs...)
    return real(inner(ρ, H; kwargs...))
end

function loss(ψ::States, H::MPO, model::AbstractVariationalCircuit; kwargs...)
    Uψ = model(ψ; kwargs...)
    return loss(Uψ, H; kwargs...)
end

function mVQE.loss(ψ::States, H::MPO, model::AbstractVariationalCircuit, samples::Int; compute_std=false, parallel=false, kwargs...)
    if parallel
        losses = ThreadsX.collect(loss(ψ, H, model; kwargs...) for _ in 1:samples)
    else
        losses = [loss(ψ, H, model; kwargs...) for _ in 1:samples]
    end

    if compute_std
        return mean(losses), std(losses) / sqrt(samples)
    else
        return mean(losses)
    end
end

function loss(ψs::VectorAbstractMPS, H::MPO; kwargs...)
    E = 0.
    for ψ in ψs
          E += loss(ψ, H; kwargs...)
    end
    return E / length(ψs)
end
 

# Property
function State_length(ψ::AbstractMPS)
    return length(ψ)
end

function State_length(ψs::VectorAbstractMPS)
    return length(ψs[1])
end

function loss_and_grad(ψs::States, H::MPO, model::AbstractVariationalCircuit; kwargs...)
    y, ∇ = withgradient(Flux.params(model)) do
        loss(ψs, H, model; kwargs...)
    end
    return [y, ∇]
end

function loss_and_grad(ψs::States, H::MPO, model::AbstractVariationalCircuit, samples::Int; kwargs...)
    g = mVQE.loss_and_grad(ψs, H, model; kwargs...)
    for _ in 1:samples - 1
        g += mVQE.loss_and_grad(ψs, H, model; kwargs...)
    end
    return g / samples

end

function get_loss_and_grad_threaded(ψs, H::mVQE.MPO; samples::Int=1, kwargs...)
    # Get the number of threads
    nthreads = Threads.nthreads()

    # If the number of threads is larger than the number of states, set the number of threads to the number of states
    if nthreads > samples
        samples_per_thread = 1
        nthreads = samples
    else
        # Split the samples into nthreads
        samples_per_thread = samples ÷ nthreads
    end

    # Use the first nthreads threads to calculate the loss
    loss_and_grad_serial_parallel(model) = loss_and_grad(ψs, H, model, samples_per_thread, kwargs...)
    loss_and_grad_threaded(model) = ThreadsX.sum(loss_and_grad_serial_parallel(model)/nthreads for i in 1:nthreads)
    return loss_and_grad_threaded
end

function get_loss_and_grad_distributed(ψs, H::mVQE.MPO; samples::Int=1, kwargs...)
    nthreads = length(workers())
            
    # If the number of threads is larger than the number of states, set the number of threads to the number of states
    if nthreads > samples
        samples_per_process = 1
        nthreads = samples
    else
        # Split the samples into nthreads
        samples_per_process = samples ÷ nthreads
    end
    @everywhere @eval  begin 
        using mVQE
    end

    sendto(workers(), ψs=ψs, H=H, samples=samples, samples_per_process=samples_per_process, kwargs=kwargs)

    @everywhere begin
        function cache(model::mVQE.AbstractVariationalCircuit)
            return mVQE.loss_and_grad(ψs, H, model, samples_per_process; kwargs...) / samples
        end
    end

    function loss_and_grad_distributed(model::mVQE.AbstractVariationalCircuit)
        return @distributed (+) for i = 1:samples
            cache(model) 
        end
    end
    return loss_and_grad_distributed
end


# Optimize with gradient descent
function optimize_and_evolve(ψs, H::MPO, model::AbstractVariationalCircuit,
                             ; optimizer=LBFGS(; maxiter=50), samples::Int=1, parallel=false, threaded=false, kwargs...)
    
    local loss_and_grad_local
    
    if parallel && samples > 1
        if  threaded # Deprecated
            println("Warning: threaded is deprecated.")
            loss_and_grad_local = get_loss_and_grad_threaded(ψs, H; samples=samples, kwargs...)
        
        else # Use Distributed
            loss_and_grad_local = get_loss_and_grad_distributed(ψs, H; samples=samples, kwargs...)
        end

    else
        loss_and_grad_serial(model) = loss_and_grad(ψs, H, model, samples, kwargs...)
        loss_and_grad_local = loss_and_grad_serial
    end
    
    model_optim, loss_v, gradient_, niter, history = optimize(loss_and_grad_local, model, optimizer)
    
    misc = Dict("loss" => loss_v, "gradient" => gradient_, "niter" => niter, "history" => history)

    return loss_v, model_optim, model_optim(ψs; kwargs...), misc
end

function optimize_and_evolve(k::Int, measurement_indices::Vector{Int}, ρ::States, H::MPO, model::AbstractVariationalCircuit
                             ;k_init=1, misc=Vector(undef, k), θs=Vector(undef, k), verbose=false, callback=(; kwargs_...) -> true, kwargs...)

    @assert k_init <= k
    @assert length(θs) == length(misc)

    if length(θs) < k
        @assert k_init != 1
        θs = [θs; Vector(undef, k - length(θs))]
        misc = [misc; Vector(undef, k - length(misc))]
    end
    
    loss_value = 1e10
    ρ, m = projective_measurement(ρ; indices=measurement_indices, reset=1)

    for i in k_init:k
        loss_value, θs[i], ρ, misc_ = optimize_and_evolve(ρ, H, model; θ=get(θs, i, initialize_circuit(model)), kwargs...)
        misc[i] = misc_

        ρ, m = projective_measurement(ρ; indices=measurement_indices, reset=1)

        if verbose
            println("iter: $i")
            println("Loss: $loss_value")
            if m !== nothing
                m = mean(mean(m)) .- 1
                println("Measured: $m")
            end
            println("")
            flush(stdout)
            flush(stderr)
        end

        out = callback(; loss_value, θs=θs[1:i], ρ, misc=misc[1:i], i=i+1)
        if out === nothing || out == true
            continue
        else
            break
        end

    end

    
    return loss_value, θs, ρ, misc
end

# To make Trea


end