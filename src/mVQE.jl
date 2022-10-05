module mVQE

using PastaQ
using ITensors
using Random
using OptimKit
using Zygote
using Statistics
using ThreadsX

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
using mVQE.Circuits: AbstractVariationalCircuit, AbstractVariationalMeasurementCircuit, generate_circuit, initialize_circuit


# Cost function
function loss(ψ::MPS, H::MPO; kwargs...)
    return inner(ψ', H, ψ; kwargs...)
end

function loss(ρ::MPO, H::MPO; kwargs...)
    return real(inner(ρ, H; kwargs...))
end

function loss(ψ::States, H::MPO, model::AbstractVariationalCircuit, θ; noise=nothing, kwargs...)
    Uψ = runcircuit(ψ, model, θ; noise, kwargs...)
    return loss(Uψ, H; kwargs...)
end

function loss(ψ::States, H::MPO, model::AbstractVariationalCircuit, θ, samples::Int; compute_std=false, noise=nothing, kwargs...)
    losses = [loss(ψ, H, model, θ; noise, kwargs...) for _ in 1:samples]

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

# Optimize with gradient descent

function optimize_and_evolve(ψs::States, H::MPO, model::AbstractVariationalCircuit
                             ;θ=initialize_circuit(model), optimizer=LBFGS(; maxiter=50), kwargs...)
    
    loss_wrapper(θ) = loss(ψs, H, model, θ; kwargs...)

    function loss_and_grad(θ)
        y, (∇,) = withgradient(loss_wrapper, θ)
        return y, ∇
    end

    θ, loss_v, gradient_, niter, history = optimize(loss_and_grad, θ, optimizer)
    
    misc = Dict("loss" => loss_v, "gradient" => gradient_, "niter" => niter, "history" => history)
    
    return loss_v, θ, runcircuit(ψs, model, θ; kwargs...), misc
end


function withgradient_vector(args...; kwargs...)
    y, (∇,) = withgradient(args...; kwargs...)
    return [y, ∇]
end

function optimize_and_evolve(ψs::States, H::MPO, model::AbstractVariationalMeasurementCircuit, samples::Int
                             ;θ=initialize_circuit(model), optimizer=LBFGS(; maxiter=50), parallel=false, kwargs...)
    
    local loss_and_grad
    if parallel
        # Get the number of threads
        nthreads = Threads.nthreads()

        # Split the samples into nthreads
        samples_per_thread = samples ÷ nthreads

        # Use the first nthreads-1 threads to calculate the loss
        loss_wrapper_parallel(θ) = loss(ψs, H, model, θ, samples_per_thread; kwargs...)
        function loss_and_grad_parallel(θ)
            GC.enable(false)
            out = ThreadsX.sum(withgradient_vector(loss_wrapper_parallel, θ)/nthreads for i in 1:nthreads)
            GC.enable(true)
            GC.gc()
            return out
        end
        loss_and_grad = loss_and_grad_parallel

    else
        loss_wrapper_(θ) = loss(ψs, H, model, θ, samples; kwargs...)
        function loss_and_grad_(θ)
            y, (∇,) = withgradient(loss_wrapper_, θ)
            return y, ∇
        end
        loss_and_grad = loss_and_grad_
    end

    θ, loss_v, gradient_, niter, history = optimize(loss_and_grad, θ, optimizer)
    
    misc = Dict("loss" => loss_v, "gradient" => gradient_, "niter" => niter, "history" => history)

    return loss_v, θ, runcircuit(ψs, model, θ; kwargs...), misc
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

end