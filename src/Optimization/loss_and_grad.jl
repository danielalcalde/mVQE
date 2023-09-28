
# Cost function
loss(ψ::AbstractMPS, H::AbstractMPS; kwargs...) = expect(ψ, H; kwargs...)

loss(ψ::States, H::PartialMPO; kwargs...) = real(expect(ψ, H; kwargs...))
loss(ψ::States, Hs::MPOVectorType; kwargs...) = sum(real(expect(ψ, H; kwargs...)) for H in Hs)
loss(ψ::States, Hs::Vector{AbstractMPS}; kwargs...) = sum(real(expect(ψ, H; kwargs...)) for H in Hs)

function loss(ψ::States, H::MPOTypes, model::AbstractVariationalCircuit; fake_measurement_regu::Number=0, kwargs...)
    Uψ = model(ψ; kwargs...)
    l = loss(Uψ, H; kwargs...)
    if fake_measurement_regu != 0
        Uψ2 = model(ψ; fake_measurement_feedback=true, kwargs...)
        l2 = loss(Uψ2, H; kwargs...)
        l_ = @Zygote.ignore min(l, 1.)
        l2 -= @Zygote.ignore l2
        l += -fake_measurement_regu * l_ * l2
    end
    return l
end

function loss(ψ::States, H::MPOTypes, model::AbstractVariationalCircuit, sample_nr::Int; compute_std=false, parallel=false, kwargs...)
    if parallel
        losses = ThreadsX.collect(loss(ψ, H, model; kwargs...) for _ in 1:sample_nr)
    else
        losses = [loss(ψ, H, model; kwargs...) for _ in 1:sample_nr]
    end

    if compute_std
        return mean(losses), std(losses) / sqrt(sample_nr)
    else
        return mean(losses)
    end
end

function loss(ψs::VectorAbstractMPS, H::MPOTypes; kwargs...)
    E = 0.
    for ψ in ψs
          E += loss(ψ, H; kwargs...)
    end
    return E / length(ψs)
end

function get_loss_and_grad(loss_func::Function; kwargs...)
    function loss_and_grad(model::AbstractVariationalCircuit)
        y, ∇ = withgradient(Flux.params(model)) do
            loss_func(model; kwargs...)
        end
        return [y, ∇]
    end
    return loss_and_grad
end

function loss_and_grad(ψs::States, H::MPOTypes, model::AbstractVariationalCircuit; kwargs...)
    y, ∇ = withgradient(Flux.params(model)) do
        loss(ψs, H, model; kwargs...)
    end
    return [y, ∇]
end


function loss_and_grad(ψs::States, H::MPOTypes, model::AbstractVariationalCircuit, sample_nr::Int; compute_std=false, kwargs...)
    losses_and_grads = [loss_and_grad(ψs, H, model; kwargs...) for _ in 1:sample_nr]

    if compute_std
        losses = [losses_and_grads[i][1] for i in 1:sample_nr]
        grads = [losses_and_grads[i][2] for i in 1:sample_nr]

        mean_grad = grads[1]
        var_grad = grads[1] * grads[1]
        for grad in grads[2:end]
            mean_grad += grad
            var_grad += grad * grad
        end
        mean_grad /= sample_nr
        var_grad /= sample_nr
        std_grad = sqrt(var_grad - mean_grad * mean_grad)

    
        return mean(losses), std(losses) / sqrt(sample_nr), mean_grad, std_grad
    else
        
        return mean(losses_and_grads[i][1] for i in 1:sample_nr), mean(losses_and_grads[i][2] for i in 1:sample_nr)
    end
end

function get_loss_and_grad_threaded(ψs, H::MPOTypes; sample_nr::Int=1, kwargs...)
    # Get the number of threads
    nthreads = Threads.nthreads()

    # If the number of threads is larger than the number of states, set the number of threads to the number of states
    if nthreads > sample_nr
        samples_per_thread = 1
        nthreads = sample_nr
    else
        # Split the sample_nr into nthreads
        samples_per_thread = sample_nr ÷ nthreads
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

function get_loss_and_grad_distributed(ψs, H::MPOTypes; sample_nr::Int=1, fix_seed=false, kwargs...)
    #nthreads = length(workers())

    @everywhere @eval Main begin
        using mVQE: loss_and_grad, AbstractVariationalCircuit
        using Random
    end
    my_secret = Random.randstring(100)
    sendto(workers(), ψs=ψs, H=H, sample_nr=sample_nr, kwargs=kwargs, secret=my_secret)
    @everywhere @eval Main begin
        function loss_and_grad_with_args(model::AbstractVariationalCircuit)
            return loss_and_grad(ψs, H, model; kwargs...)
        end
    end
    local fixed_seed
    if fix_seed
        fixed_seed = rand(UInt)
    end
    function loss_and_grad_distributed(model::AbstractVariationalCircuit; get_list=false)
        if fix_seed
            seed = fixed_seed
        else
            seed = rand(UInt)
        end
        
        if get_list
            op = vcat
        else
            op = +
        end
        sendto(workers(), model=model)
        out = @distributed (op) for i = 1:sample_nr
            @assert Main.secret == my_secret "The secret does not match, this might happen if get_loss_and_grad_distributed is called twice."
            Random.seed!(seed + i)
            Main.loss_and_grad(Main.ψs, Main.H, Main.model; Main.kwargs...)
        end
        
        if get_list
            out = reshape(out, 2, sample_nr)
            l = out[1, :]
            grads = [fix_grads(grad, model) for grad in out[2, :]]
        else
            l, grads = out[1]/ sample_nr, fix_grads(out[2], model)/ sample_nr
        end

        return l, grads
    end
    return loss_and_grad_distributed
end

function get_loss_distributed(ψs, H::MPOTypes; sample_nr::Int=1, fix_seed=false, kwargs...)
            
    @everywhere @eval Main begin
        using mVQE: loss, AbstractVariationalCircuit
        using Random
    end
    my_secret = Random.randstring(100)
    sendto(workers(), ψs=ψs, H=H, sample_nr=sample_nr, kwargs=kwargs, loss_secret=my_secret)
    @everywhere @eval Main begin
        function loss_with_args(model::AbstractVariationalCircuit)
            return loss(ψs, H, model; kwargs...) / sample_nr
        end
    end
    local fixed_seed
    if fix_seed
        fixed_seed = rand(UInt)
    end

    function loss_distributed(model::AbstractVariationalCircuit)
        if fix_seed
            seed = fixed_seed
        else
            seed = rand(UInt)
        end

        l = @distributed (+) for i = 1:sample_nr
            @assert Main.loss_secret == my_secret "The secret does not match, this might happen if get_loss_and_grad_distributed is called twice."
            Random.seed!(seed + i)
            Main.loss_with_args(model)
        end
        return l
    end
    return loss_distributed
end

function get_loss_and_grad(ψs, H::MPOTypes; sample_nr::Int=1, parallel=false, fix_seed=false, threaded=false, kwargs...)
    local loss_and_grad_local
    
    if parallel && sample_nr > 1
        if  threaded # Deprecated
            println("Warning: threaded is deprecated.")
            loss_and_grad_local = get_loss_and_grad_threaded(ψs, H; sample_nr, kwargs...)
        
        else # Use Distributed
            loss_and_grad_local = get_loss_and_grad_distributed(ψs, H; sample_nr, fix_seed, kwargs...)
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
            return loss_and_grad(ψs, H, model, sample_nr; kwargs...)
        end
        loss_and_grad_local = loss_and_grad_serial
    end
    return loss_and_grad_local
end