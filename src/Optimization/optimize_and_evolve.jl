
# Optimize with gradient descent
function optimize_and_evolve(ψs::States, H::MPOTypes, model::AbstractVariationalCircuit;
    optimizer=LBFGS(; maxiter=50), sample_nr::Int=1, parallel=false,
    threaded=false, callback=callback_,
    fix_seed=false, kwargs_optim=Dict(),
    loss_and_grad=nothing,
    kwargs...)

    if loss_and_grad === nothing
        loss_and_grad = get_loss_and_grad(ψs, H; sample_nr, parallel, fix_seed, threaded, kwargs...)
    end

    model_optim, loss_v, misc = optimize(loss_and_grad, model, optimizer, callback; kwargs_optim...)

    return loss_v, model_optim, model_optim(ψs; kwargs...), misc
end

function optimize_and_evolve(k::Int, measurement_indices::Vector{<:Integer}, ρ::States, H::MPOTypes, model::AbstractVariationalCircuit
    ; k_init=1, misc=Vector(undef, k), θs=Vector(undef, k), verbose=false,
    callback=(; kwargs_...) -> true, (finalize!)=OptimKit._finalize!,
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

        out = callback(; loss_value, θs=θs[1:i], ρ, misc=misc[1:i], i=i + 1)
        if out === nothing || out == true
            continue
        else
            break
        end

    end


    return loss_value, θs, ρ, misc
end