module StateFactory
using ITensors
using PastaQ
using LinearAlgebra

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

function infinite_temp_MPO(hilbert; normalize=true, eltype=Complex{Float64})
    N = length(hilbert)
    ρ = MPO(N)
    @assert N > 1

    if normalize
        norm = 0.5
    else
        norm = 1.
    end

    # First Index
    s = hilbert[1]
    sₚ = prime(s)
    link_right = Index(1, "Link,l=1")
    t = ITensor(eltype, link_right, s, sₚ)
    t[link_right => 1, s => 1, sₚ => 1] = norm
    t[link_right => 1, s => 2, sₚ => 2] = norm
    ρ[1] = t

    for i in 2:N-1
        s = hilbert[i]
        sₚ = prime(s)
        link_left = link_right
        link_right = Index(1, "Link,l=$i")
        t = ITensor(eltype, link_left, link_right, s, sₚ)
        t[link_left => 1, link_right => 1, s => 1, sₚ => 1] = norm
        t[link_left => 1, link_right => 1, s => 2, sₚ => 2] = norm
        ρ[i] = t
    end
    
    # Last Index
    s = hilbert[N]
    sₚ = prime(s)
    t = ITensor(link_right, s, sₚ)
    t[link_right => 1, s => 1, sₚ => 1] = norm
    t[link_right => 1, s => 2, sₚ => 2] = norm
    ρ[N] = t

    return ρ
end

function AKLT_half_tensor(link_1, physical, link_2; location=1, spin_selection=nothing)
    if location == 1
        @assert link_1.space == 2 && link_2.space == 4 "link_1.space = $(link_1.space), link_2.space = $(link_2.space)"
    else
        @assert link_2.space == 2 && link_1.space == 4 "link_1.space = $(link_1.space), link_2.space = $(link_2.space)"
    end
    tensor = ITensor(link_1, physical, link_2)
    
    if location == 1
        
        tensor[1, 1, 1] = -1/sqrt(2)
        tensor[1, 1, 2] = -1/sqrt(2)
        tensor[1, 2, 4] = 1.0
        tensor[2, 1, 3] = -1.0
        tensor[2, 2, 1] = 1/sqrt(2)
        tensor[2, 2, 2] = -1/sqrt(2)

    elseif location == 2

        tensor[1, 1, 2] = 0.75
        tensor[1, 2, 1] = 0.75
        tensor[2, 1, 2] = 0.25
        tensor[2, 2, 1] = -0.25
        tensor[3, 2, 2] = -0.3535533905932738
        tensor[4, 1, 1] = -0.3535533905932738

    else
        @assert false, "location parameter needs to be 1, 2 not $location"
    end
    
    if spin_selection !== nothing
        if location == 1
            tensor_s = ITensor(link_1)
        else
            tensor_s = ITensor(link_2)
        end
        tensor_s[1] = spin_selection[1]
        tensor_s[2] = spin_selection[2]
        tensor = tensor * tensor_s
    end
    
    return tensor
end

function AKLT_half_tensor_girvin(link_1, physical, link_2; location=1, spin_selection=nothing)
    # notebooks/workprojects/tensornetworks/theory/AKLT%20spin%20Half.ipynb
    # |+> = |10>
    # |-> = |01>
    # |0> = |00>
    if location == 1
        @assert link_1.space == 2 && link_2.space == 3 "link_1.space = $(link_1.space), link_2.space = $(link_2.space)"
    else
        @assert link_2.space == 2 && link_1.space == 3 "link_1.space = $(link_1.space), link_2.space = $(link_2.space)"
    end
    tensor = ITensor(link_1, physical, link_2)
    
    if location == 1
        
        tensor[1, 1, 3] = 1.0
        tensor[1, 2, 1] = -0.577350269189626
        tensor[1, 2, 2] = 0.816496580927726
        tensor[2, 1, 1] = 0.8164965809277263
        tensor[2, 1, 2] = 0.577350269189626

    elseif location == 2

        tensor[1, 1, 2] = 0.8164965809277264
        tensor[1, 2, 1] = 0.577350269189626
        tensor[2, 1, 2] = -0.28867513459481303
        tensor[2, 2, 1] = 0.40824829046386324
        tensor[3, 1, 1] = -0.5000000000000001

    else
        @assert false, "location parameter needs to be 1, 2 not $location"
    end
    
    if spin_selection !== nothing
        if location == 1
            tensor_s = ITensor(link_1)
        else
            tensor_s = ITensor(link_2)
        end
        tensor_s[1] = spin_selection[1]
        tensor_s[2] = spin_selection[2]
        tensor = tensor * tensor_s
    end
    
    return tensor
end

function AKLT_half(spin1_vec, spin2_vec, hilbert; basis="girvin")
    if basis == "clebsch"
        tensor_gen = AKLT_half_tensor
        middle_index_length = 4

    elseif basis == "girvin"
        tensor_gen = AKLT_half_tensor_girvin
        middle_index_length = 3

    else
        @assert false, "basis needs to be clebsh or girvin not $basis"
    end

    L2 = length(hilbert)
    L = Int(L2 / 2)
    tensors = Vector{ITensor}(undef, L2)
    
    l_l = Index(2,"Link,n=0")
    l_r = Index(middle_index_length,"Link,n=1")
    tensors[1] = tensor_gen(l_l, hilbert[1], l_r, location=1, spin_selection=spin1_vec)
    
    l_l = l_r
    l_r = Index(2,"Link,n=2")
    tensors[2] = tensor_gen(l_l, hilbert[2], l_r, location=2)
    
    for i in 2: L - 1
        l_l = l_r
        l_r = Index(middle_index_length,"Link,n=$(2i-1)")
        tensors[2*i - 1] = tensor_gen(l_l, hilbert[2*i-1], l_r, location=1)
        
        l_l = l_r
        l_r = Index(2,"Link,n=$(2i)")
        tensors[2*i] = tensor_gen(l_l, hilbert[2*i], l_r, location=2)
    end
    l_l = l_r
    l_r = Index(middle_index_length,"Link,n=$(2 * L -1)")
    tensors[2 * L - 1] = tensor_gen(l_l, hilbert[2*L-1], l_r, location=1)
    
    l_l = l_r
    l_r = Index(2,"Link,n=$(2 * L)")
    tensors[2 * L] = tensor_gen(l_l, hilbert[2*L], l_r, location=2, spin_selection=spin2_vec)
    mps = MPS(tensors)
    orthogonalize!(mps, 1)
    mps[1] /= norm(mps[1])
    #truncate!(mps)
    return mps
end

function orthogonalize(ψ2, ψ1)
    r = -inner(ψ1, ψ2)
    ψ2n = +(ψ2, r * ψ1; cutoff=1e-10)
    ψ2n = ψ2n/norm(ψ2n)
    return ψ2n
end

function orthogonalize_basis(ψs::Vector{MPS})
    ψs = copy(ψs)
    for i in 1:length(ψs)
        for j in 1:i-1
            @show i, j
            r = inner(ψs[i], ψs[j])
            ψs[i] = orthogonalize(ψs[i], ψs[j])
        end
    end
    return ψs
end

"""
Constructs the 4 AKLT states.
"""
function AKLT_halfs(hilbert; orthogonalize=false, kwargs...)
    aklts = [AKLT_half([1, 0], [1, 0], hilbert; kwargs...),
            AKLT_half([1, 0], [0, 1], hilbert; kwargs...),
            AKLT_half([0, 1], [1, 0], hilbert; kwargs...),
            AKLT_half([0, 1], [0, 1], hilbert; kwargs...)]

    if orthogonalize
        r = inner(aklts[1], aklts[4])
        aklts[1] = +(aklts[1], (-r) * aklts[4]; cutoff=1e-10)
        aklts[1] =  aklts[1]/norm(aklts[1])
    end
    return aklts
end

# W state
function W_state_tensor(l, lmax, link1, ph, link2)
    if l == 1
        tensor = ITensor(ph, link2)
        tensor[:] = (I(2).*1.)[:]
    elseif l==lmax
        tensor = ITensor(link1, ph)
        tensor[2, 1] = sqrt((l-1.) / l)
        tensor[1, 2] = sqrt(1. / l)
    else
        tensor = ITensor(link1, ph, link2)
        tensor[1, 1, 1] = 1.
        tensor[2, 1, 2] = sqrt((l-1.) / l)
        tensor[1, 2, 2] = sqrt(1. / l)
        
    end
    return tensor
end

function W_state(hilbert)
    N = length(hilbert)
    links = [Index(2, "Link,n=$i") for i in 1:N]
    tensors = [W_state_tensor(li, N, links[mod1(li-1, N)], hilbert[li], links[mod1(li, N)]) for li in 1:N]
    ψ = MPS(tensors)
    orthogonalize!(ψ, 1)
    ψ[1] /= norm(ψ[1])
    return ψ
end


# end module
end