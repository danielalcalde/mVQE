module StateFactory
using ITensors
using PastaQ

function random_MPS(hilbert, k; ancilla_indices=nothing)
    N = length(hilbert)

    if ancilla_indices isa Int
        ancilla_indices = [ancilla_indices]
    end

    ψs = Vector{MPS}(undef,k)
    states = Vector{Vector{Int}}(undef,k)
    for i in 1:k
        state = rand(0:1, N)

        if ancilla_indices !== nothing
            state[ancilla_indices] .= 0
        end

        states[i] = state
        ψs[i] = productstate(hilbert, state)
    end
    return ψs, states
end

function infinite_temp_MPO(hilbert)
    N = length(hilbert)
    ρ = MPO(N)
    @assert N > 1

    # First Index
    s = hilbert[1]
    sₚ = prime(s)
    link_right = Index(1, "Link,l=1")
    t = ITensor(link_right, s, sₚ)
    t[link_right => 1, s => 1, sₚ => 1] = 0.5 + 0.0im
    t[link_right => 1, s => 2, sₚ => 2] = 0.5 + 0.0im
    ρ[1] = t

    for i in 2:N-1
        s = hilbert[i]
        sₚ = prime(s)
        link_left = link_right
        link_right = Index(1, "Link,l=$i")
        t = ITensor(link_left, link_right, s, sₚ)
        t[link_left => 1, link_right => 1, s => 1, sₚ => 1] = 0.5 + 0.0im
        t[link_left => 1, link_right => 1, s => 2, sₚ => 2] = 0.5 + 0.0im
        ρ[i] = t
    end
    
    # Last Index
    s = hilbert[N]
    sₚ = prime(s)
    t = ITensor(link_right, s, sₚ)
    t[link_right => 1, s => 1, sₚ => 1] = 0.5 + 0.0im
    t[link_right => 1, s => 2, sₚ => 2] = 0.5 + 0.0im
    ρ[N] = t

    return ρ
end

# end module
end