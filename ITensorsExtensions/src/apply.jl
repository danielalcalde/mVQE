apply_nogategrad(gates::Vector{ITensor}, ψ::Union{MPS,MPO}; kwargs...) = apply(gates, ψ; kwargs...)

function Zygote.rrule(
  ::typeof(apply_nogategrad), gates::Vector{ITensor}, ψ::Union{MPS,MPO}; apply_dag=false, kwargs...
)
  function apply_pullback(ψbar::Union{MPS,MPO})
        gates_dag = [swapprime(dag(gate), 0 => 1) for gate in gates[end:-1:1]]
        x̄2 = apply(gates_dag, ψbar; apply_dag=apply_dag, kwargs...)
    return (Zygote.NoTangent(), Zygote.NoTangent(), x̄2)
  end

  return apply(gates, ψ; apply_dag=apply_dag, kwargs...), apply_pullback
end

function findsite(gate::ITensor, ψ)
    for (i, ψi) in enumerate(ψ)
        if commonind(gate, ψi) !== nothing
            return i
        end
    end
end

function sort_gates(gates, ψ::Union{MPS,MPO}; unsort_func=false)
    sorted_gates = Vector{Union{Nothing, ITensor}}(undef, length(ψ))
    sorted_gates .= fill(nothing, length(ψ))
        
    order = Vector{Int}(undef, length(gates))
            
    for (i, gate) in enumerate(gates)
        k = findsite(gate, ψ)
        if k === nothing
            error("ka")
        end
        if sorted_gates[k] !== nothing
            error("There is two gates acting on the same lattice sizes $k.")
        end
        order[i] = k
        sorted_gates[k] = gate
    end
    
    if unsort_func
       return sorted_gates, sorted_gates -> [sorted_gates[i] for i in order] 
    end
    
    return sorted_gates
end


function apply_onequbit(gates::Vector{ITensor}, ψ::MPS; get_sorted_gates=false, unitary=false, kwargs...)
    gates, unsort_func = sort_gates(gates, ψ; unsort_func=true)
    ψtensors = Vector{ITensor}(undef, length(ψ))
    for (i, (gate, ψi)) in enumerate(zip(gates, ψ))
        if gate === nothing
            ψtensors[i] = ψi
        else
            index = commonind(gate, ψi)
            ψtensors[i] = noprime(ψi * gate, index')
        end
    end
    if unitary
        ψo = MPS(ψtensors, ψ.llim, ψ.rlim)
    else
        ψo = MPS(ψtensors)
    end
    if get_sorted_gates
        return ψo, gates, unsort_func
    end
    return ψo
end

function Zygote.rrule(
  ::typeof(apply_onequbit), gates::Vector{ITensor}, ψ::MPS; apply_dag=false, kwargs...
)
  ψo, gates_sorted, unsort_func = apply_onequbit(gates, ψ; get_sorted_gates=true)

  function apply_pullback(ψbar::Union{MPS,MPO})
        N = length(ψ) 
        ψo_dag = conj(ψo)
        ψ_dag = conj(ψ)
        
        env_L = get_enviroments(ψbar, ψo_dag, false)
        env_R = get_enviroments(ψbar, ψo_dag, true)
        
        gates_out_bar_sorted = Vector{ITensor}(undef, length(ψ))
        for (i, gate) in enumerate(gates_sorted)

            if gate !== nothing
                site = commonind(ψ[i], ψbar[i])
                ψbar_i = prime(ψbar[i], site)
                if i == 1
                    gates_out_bar_sorted[i] = env_R[2] * ψ_dag[i] * ψbar_i
                elseif i == N
                    gates_out_bar_sorted[i] = env_L[end-1] * ψ_dag[i] * ψbar_i
                else
                    gates_out_bar_sorted[i] = env_L[i-1] * ψ_dag[i] * ψbar_i * env_R[i + 1]
                end
            end
        end
        gates_out_bar = unsort_func(gates_out_bar_sorted)
    
        # Derivative of wave funciton
        gates_dag = [swapprime(dag(gate), 0 => 1) for gate in gates[end:-1:1]]
        ψ_out_bar = apply_onequbit(gates_dag, ψbar)
        
        return (Zygote.NoTangent(), gates_out_bar, ψ_out_bar)
      end

  return apply(gates, ψ; apply_dag=apply_dag, kwargs...), apply_pullback
end