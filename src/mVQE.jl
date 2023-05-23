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

include("FluxExtensions.jl")
include("pyflexmps.jl")
include("Optimizers.jl")
include("Misc.jl")
include("Hamiltonians.jl")

include("ITensorsExtension.jl")

include("StateFactory.jl")

include("Gates.jl")
include("Layers.jl")
include("Circuits.jl")
include("GirvinProtocol.jl")

using ..ITensorsExtension: VectorAbstractMPS, States, projective_measurement
using ..Circuits: AbstractVariationalCircuit, AbstractVariationalMeasurementCircuit, generate_circuit
using ..Optimizers: callback_, optimize

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

function loss(ψ::States, H::MPO, model::AbstractVariationalCircuit, samples::Int; compute_std=false, parallel=false, kwargs...)
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
    g = loss_and_grad(ψs, H, model; kwargs...)
    for _ in 1:samples - 1
        g += loss_and_grad(ψs, H, model; kwargs...)
    end
    return g / samples

end

function get_loss_and_grad_threaded(ψs, H::MPO; samples::Int=1, kwargs...)
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

function fix_grads(grads::Zygote.Grads, model::T) where {T}
    @assert hasmethod(Flux.params, (T,)) "The model does not have a params method"
    params = Flux.params(model)
    d = IdDict()
    for i in 1:length(params)
        @assert grads.params[i] == params[i] "The params of the model and the grads do not match"
        d[params[i]] = grads[grads.params[i]]
    end
    return Zygote.Grads(d, params)
end

function get_loss_and_grad_distributed(ψs, H::MPO; samples::Int=1, fix_seed=false, kwargs...)
    #nthreads = length(workers())
            
    @everywhere @eval Main begin
        using mVQE: loss_and_grad, AbstractVariationalCircuit
        using Random
    end
    my_secret = Random.randstring(100)
    sendto(workers(), ψs=ψs, H=H, samples=samples, kwargs=kwargs, secret=my_secret)
    @everywhere @eval Main begin
        function loss_and_grad_with_args(model::AbstractVariationalCircuit)
            return loss_and_grad(ψs, H, model; kwargs...) / samples
        end
    end
    local fixed_seed
    if fix_seed
        fixed_seed = rand(UInt)
    end

    function loss_and_grad_distributed(model::AbstractVariationalCircuit)
        if fix_seed
            seed = fixed_seed
        else
            seed = rand(UInt)
        end

        l, grads = @distributed (+) for i = 1:samples
            @assert Main.secret == my_secret "The secret does not match, this might happen if get_loss_and_grad_distributed is called twice."
            Random.seed!(seed + i)
            Main.loss_and_grad_with_args(model)
        end
        return l, fix_grads(grads, model)
    end
    return loss_and_grad_distributed
end


# Optimize with gradient descent
function optimize_and_evolve(ψs::States, H::MPO, model::AbstractVariationalCircuit;
                             optimizer=LBFGS(; maxiter=50), samples::Int=1, parallel=false,
                             threaded=false, callback=callback_,
                             fix_seed=false, kwargs_optim=Dict(),
                             kwargs...)
    
    local loss_and_grad_local
    
    if parallel && samples > 1
        if  threaded # Deprecated
            println("Warning: threaded is deprecated.")
            loss_and_grad_local = get_loss_and_grad_threaded(ψs, H; samples, kwargs...)
        
        else # Use Distributed
            loss_and_grad_local = get_loss_and_grad_distributed(ψs, H; samples, fix_seed, kwargs...)
        end

    else
        local seed
        if fix_seed
            seed = rand(UInt)
        end
        function loss_and_grad_serial(model)
            if fix_seed
                Random.seed!(seed)
            end
            return loss_and_grad(ψs, H, model, samples; kwargs...)
        end
        loss_and_grad_local = loss_and_grad_serial
    end
    model_optim, loss_v, gradient_, niter, history = optimize(loss_and_grad_local, model, optimizer, callback; kwargs_optim...)
    
    misc = Dict("loss" => loss_v, "gradient" => gradient_, "niter" => niter, "history" => history)

    return loss_v, model_optim, model_optim(ψs; kwargs...), misc
end

function optimize_and_evolve(k::Int, measurement_indices::Vector{Int}, ρ::States, H::MPO, model::AbstractVariationalCircuit
                             ;k_init=1, misc=Vector(undef, k), θs=Vector(undef, k), verbose=false,
                             callback=(; kwargs_...) -> true, finalize! = OptimKit._finalize!,
                             kwargs...)

    @assert k_init <= k
    @assert length(θs) == length(misc)

    if length(θs) < k
        @assert k_init != 1
        θs = [θs; Vector(undef, k - length(θs))]
        misc = [misc; Vector(undef, k - length(misc))]
    end
    
    loss_value = 1e10
    ρ = projective_measurement(ρ; indices=measurement_indices, reset=1)

    for i in k_init:k
        loss_value, θs[i], ρ, misc_ = optimize_and_evolve(ρ, H, model; θ=get(θs, i, initialize_circuit(model)), kwargs...)
        misc[i] = misc_

        ρ = projective_measurement(ρ; indices=measurement_indices, reset=1)

        if verbose
            println("iter: $i")
            println("Loss: $loss_value")
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