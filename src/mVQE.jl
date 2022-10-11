module mVQE

using PastaQ
using ITensors
using Random
using OptimKit
using Zygote
using Statistics
using ThreadsX
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

function loss(ψ::States, H::MPO, model::AbstractVariationalCircuit; noise=nothing, kwargs...)
    Uψ = model(ψ; noise, kwargs...)
    return loss(Uψ, H; kwargs...)
end

function loss(ψ::States, H::MPO, model::AbstractVariationalCircuit, samples::Int; compute_std=false, noise=nothing, kwargs...)
    losses = [loss(ψ, H, model; noise, kwargs...) for _ in 1:samples]

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

function loss_and_grad(ψs::States, H::MPO, model::AbstractVariationalCircuit; samples::Int=1, kwargs...)
    y, ∇ = withgradient(Flux.params(model)) do
        loss(ψs, H, model, samples; kwargs...)
    end
    return [y, ∇]
end

# Optimize with gradient descent
function optimize_and_evolve(ψs, H::MPO, model::AbstractVariationalCircuit,
                             ; optimizer=LBFGS(; maxiter=50), samples::Int=1, parallel=false, nthreads=0, kwargs...)
    
    local loss_and_grad_local
    
    if parallel && samples > 1 && nthreads > 1
        # Get the number of threads
        if nthreads == 0
            nthreads = Threads.nthreads()
        end

        # If the number of threads is larger than the number of states, set the number of threads to the number of states
        if nthreads > samples
            samples_per_thread = 1
            nthreads = samples
        else
            # Split the samples into nthreads
            samples_per_thread = samples ÷ nthreads
        end

        # Use the first nthreads threads to calculate the loss
        loss_and_grad_serial_parallel(model) = loss_and_grad(ψs, H, model; samples=samples_per_thread, kwargs...)
        loss_and_grad_parallel(model) = ThreadsX.sum(loss_and_grad_serial_parallel(model)/nthreads for i in 1:nthreads)
        
        loss_and_grad_local = loss_and_grad_parallel

    else
        loss_and_grad_serial(model) = loss_and_grad(ψs, H, model; samples=samples, kwargs...)
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