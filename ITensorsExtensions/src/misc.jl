function cut_entropy(psi::MPS, b::Int)
    s = siteinds(psi)  
    orthogonalize!(psi, b)
    _,S = svd(psi[b], (linkind(psi, b-1), s[b]))
    SvN = 0.0
    for n in 1:dim(S, 1)
      p = S[n,n]^2
      SvN -= p * log(p .+ 1e-15)
    end
    return SvN
end

cut_entropy(ψ::MPS) = [cut_entropy(ψ, i) for i in 2:length(ψ)-1]

function exact_entropy(ρ)
    hilbert = [siteindex(ρ, i) for i in 1:length(ρ)]
    ρ_d = contract(ρ)
    C = combiner(hilbert)
    ρ_d = ρ_d * C * prime(C);
    _, S, _ = svd(ρ_d, ρ_d.tensor.inds[1])
    p_exact = S.tensor.storage
    return -sum(p_exact.*log2.(abs.(p_exact) .+ 1e-15))
end

function exact_entropy(ψ, indices; mem=Dict())
    if ! (hash(indices) in keys(mem))
        if length(indices) < length(ψ) - length(indices)
            indices = setdiff(collect(1:length(ψ)), indices)
        end
        ρA = tr(ψ, indices)
        mem[hash(indices)] = exact_entropy(ρA)
    else
            #@show indices
    end
    return mem[hash(indices)]
end

function mutual_info(ψinter, A, B, rest=nothing; mem=Dict())
    if rest === nothing
        rest = collect(1:length(ψinter))
        rest = setdiff(rest, union(A, B))
    end
    SAB = exact_entropy(ψinter, rest; mem)
    SA = exact_entropy(ψinter, sort(vcat(rest, B)); mem)
    SB = exact_entropy(ψinter, sort(vcat(rest, A)); mem)
    return SA + SB - SAB
end

function mutual_info_matrix(ψ, indices=collect(1:length(ψ)); mem=Dict(), verbose=false)
    l1 = length(indices)
    M = zeros(l1, l1)
    for ii in 1:l1
        if verbose
            @show ii
            flush(stdout)
        end
        i = indices[ii]

        for jj in 1:l1
            j = indices[jj]
            if i != j
                A = [i]
                B = [j]
                rest = collect(1:length(ψ))
                rest = setdiff(rest, union(A, B))
                mm = mutual_info(ψ, A, B, rest; mem)
                M[ii, jj] = mm
            end
        end
    end
    return M
end
function exact_entanglement_entropy(ψ, hilbert, state_indices)
    state_indices = sort(state_indices)
    ITensors.disable_warn_order()
    other_indices = collect(1:length(hilbert))
    deleteat!(other_indices, state_indices)
    if length(other_indices) > length(state_indices)
        other_indices, state_indices = state_indices, other_indices
    end
    
    hilbert_ancillas = hilbert[other_indices]
    ρ = tr(ψ, state_indices)
    ρ_d = contract(ρ)
    C = combiner(hilbert_ancillas)
    ρ_d = ρ_d * C * prime(C);
    _, S, _ = svd(ρ_d, ρ_d.tensor.inds[1])
    p_exact = S.tensor.storage;
    return -sum(log2.(p_exact) .* p_exact)
end