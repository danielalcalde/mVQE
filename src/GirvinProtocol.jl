module GirvinProtocol

using ITensors
using Flux
using Zygote
using TensorOperations
using Random, RandomMatrices

using ..Circuits: AbstractVariationalCircuit, AbstractVariationalMeasurementCircuit, generate_circuit, VariationalMeasurementMCFeedback
using ..Misc: get_ancilla_indices
using ..FluxExtensions: SpinTokenizer
using mVQE: Circuits, Layers

struct GirvinCircuit <: AbstractVariationalCircuit
    params::Matrix{Float64}
    GirvinCircuit(params::Matrix{Float64}) = new(params)
    GirvinCircuit(N::Int) = new(2π .* rand(N, 8))
    GirvinCircuit() = new(Matrix{Float64}(undef, 0, 0)) # Empty circuit to be used as a placeholder
end
Flux.@functor GirvinCircuit
Flux.trainable(a::GirvinCircuit) = (a.params,)
Circuits.get_N(a::GirvinCircuit) = size(a.params, 1) * 6
Circuits.get_depth(a::GirvinCircuit) = size(a.params, 2)


function GirvinCircuitIdeal(N_state::Int)
    θ_1 = 2 * atan(-1 / sqrt(2))
    θ_2 = 2 * atan(sqrt(2))
    params = hcat(fill([pi, -pi/2, θ_1, θ_2, -pi, θ_1, θ_2, -pi], Int(N_state/4))...)'
    return GirvinCircuit(Array(params))
end

function Circuits.generate_circuit(model::GirvinCircuit; params=nothing)
    if params === nothing
        params = model.params
    end
    N_state = size(params, 1) * 4
    N = size(params, 1) * 6

    gates = Vector()
    for (i, site) in enumerate(3:6:N)
        gates = vcat(gates, singlet_gate_p(site, site + 1, params[i, 1:2]...))
    end
    for (i, site) in enumerate(1:6:N)
            gates = vcat(gates, U2_gate_p(site, site+1, site+2, site+3, site+4, site+5, params[i, 3:end]...))
    end
    
    for site in 6:6:N-1
        gates = vcat(gates, bell_gate(site+1, site))
    end
    return gates
end

singlet_gate_p(i1, i2, θ_1, θ_2) = [("Rx", i1, (θ=θ_1,)), ("Ry", i2, (θ=θ_2,)), ("CX", (i2, i1))]

U_gate_L_p(c_in, i1, i2, θ_1, θ_2, ϕ_1) =[("CX", (c_in, i1)), ("CRy", (i1, c_in), (θ = θ_1,)), ("Rx", i1, (θ=ϕ_1,)),
                        ("CRy", (i1, i2), (θ = θ_2,)), ("X", i1),
                        ("CX", (c_in, i1)), ("CX", (i1, i2)), ("CX", (i2, i1))]

U_gate_R_p(c_in, i1, i2, θ_1, θ_2, ϕ_1) =[("CX", (c_in, i1)), ("CRy", (i1, c_in), (θ = θ_1,)), ("Rx", i1, (θ=ϕ_1,)),
                        ("CRy", (i1, i2), (θ = θ_2,)), ("X", i1),
                        ("CX", (c_in, i1)), ("CX", (i1, i2)), ("CX", (i2, i1)), ("SWAP", c_in, i1)]

U2_gate_p(i0, i1, i2, i3, i4, i5, θ_1, θ_2, ϕ_1, θ_3, θ_4, ϕ_2) = vcat(U_gate_L_p(i2, i1, i0, θ_1, θ_2, ϕ_1), U_gate_R_p(i3, i4, i5, θ_3, θ_4, ϕ_2))
bell_gate(i0, i1) = [("CX", (i1, i0)), ("H", i1)]


# Correct the errors the circuit introduces
struct GirvinCorrCircuit <: AbstractVariationalCircuit
    params::Matrix{Float64}
    GirvinCorrCircuit(params::Matrix{Float64}) = new(params)
    GirvinCorrCircuit(N::Int) = new(2π .* rand(N, 4))
    GirvinCorrCircuit() = new(Matrix{Float64}(undef, 0, 0)) # Empty circuit to be used as a placeholder
end
Flux.@functor GirvinCorrCircuit
Flux.trainable(a::GirvinCorrCircuit) = ()

Base.size(model::GirvinCorrCircuit) = size(model.params)
Base.size(model::GirvinCorrCircuit, i::Int) = size(model.params, i)
Circuits.get_N(a::GirvinCorrCircuit) = size(a.params, 1) * 4
Circuits.get_depth(a::GirvinCorrCircuit) = 6


function Circuits.generate_circuit(::GirvinCorrCircuit; params=nothing)
    @assert params !== nothing
    @assert size(params, 2) == 4
    N = size(params, 1)
    N_state = N * 4
    
    state_indices, = Zygote.@ignore get_ancilla_indices(N_state, [false, true, true, true, true, false])
    gates = Vector()
    
    for site in 1:N
        p = (α=params[site, 1], β=params[site, 2], γ=params[site, 3], δ=params[site, 4])
        gates = vcat(gates, [("U_girvin_rot", (state_indices[2site-1], state_indices[2site]), p)])
    end
    return gates
end

function ITensors.op(::OpName"U_girvin_rot", ::SiteType"Qubit"; α::Number, β::Number, γ::Number, δ::Number) 
    return [
    1 0 0 0
    0 exp(1im*(α-β-δ)) * cos(γ) -exp(1im*(α-β+δ)) * sin(γ)  0
    0 exp(1im*(α+β-δ)) * sin(γ)  exp(1im*(α+β+δ)) * cos(γ)  0
    0 0 0 1]
end

compare(X, Y) = sum(abs.(X - Y)) < 1e-5
compare(X, Y, tX, tY) = (compare(X, tX) && compare(Y, tY)) || (compare(X, tY) && compare(Y, tX))

function add(a, b)
    if compare(a, b)
        out = zeros(4)

    elseif compare(a, [0,0,0,0]) || compare(b, [0,0,0,0])
        out = a .+ b

    elseif compare(a, b, [pi/2,pi/2,pi/2,0], [pi/2,pi/2,pi/2,pi])
        out = [0,0,0,pi]

    elseif compare(a, b, [pi/2,pi/2,pi/2,0], [0,0,0,pi])
        out = [pi/2,pi/2,pi/2,pi]

    elseif compare(a, b, [0,0,0,pi], [pi/2,pi/2,pi/2,pi])
        out = [pi/2,pi/2,pi/2,0]
    else
        @assert false "$a $b"
    end
    return out
end

param_correction_gates(M) = GirvinCorrectionNetwork()(M) # legacy

struct GirvinCorrectionNetwork end
Flux.@functor GirvinCorrectionNetwork
Flux.trainable(::GirvinCorrectionNetwork) = ()

function (t::GirvinCorrectionNetwork)(M::Vector{T}) where T <: Real
    Zygote.@ignore begin
        correction_gates_params = Dict()  
        correction_gates_params[Vector{T}([0, 0])] = [pi/2, pi/2, pi/2, 0]
        correction_gates_params[Vector{T}([0, 1])] = [0, 0, 0, pi]
        correction_gates_params[Vector{T}([1, 0])] = [pi/2, pi/2, pi/2, pi]
        correction_gates_params[Vector{T}([1, 1])] = [0, 0, 0, 0]

        Ns_spin1 = length(M)
        M = M[2:end-1]
        
        @assert mod(length(M), 2) == 0 "M must have even length (got $(length(M)))"
        
        M = reshape(M, (2, Int(length(M)/2)))'
        angles = zeros((Ns_spin1, 4))
        for i in 1:size(M, 1)
            g = correction_gates_params[M[i, :]]
            for j in 1:2i
                angles[j, :] = add(angles[j, :], g)
            end
        end
        return angles
    end
end

function (t::GirvinCorrectionNetwork)(M::Matrix{T}) where T <: Real
    return Zygote.@ignore begin
        L, k = size(M)
        y_tar = zeros(L, 4, k)
        for i in 1:k
            y_tar[:, :, i] = t(M[:, i])
        end
        return y_tar
    end
end

###

struct GirvinCorrectionNetworkLearnable
    A::Array{T, 3} where T
    C::Matrix{T} where T
    trainable::Vector{Bool}
end
Flux.@functor GirvinCorrectionNetworkLearnable
function Flux.trainable(self::GirvinCorrectionNetworkLearnable)
    t = []
    if self.trainable[1]
        push!(t, self.A)
    end
    if self.trainable[2]
        push!(t, self.C)
    end
    return Tuple(t)
end


function GirvinCorrectionNetworkLearnable()
    A = zeros(4,4,4)
    A[1, 1, 4] = 1
    A[1, 2, 3] = 1
    A[1, 3, 2] = 1
    A[1, 4, 1] = 1
    A[2, 4, 3] = 1
    A[2, 2, 1] = 1
    A[2, 3, 4] = 1
    A[2, 1, 2] = 1
    A[3, 4, 2] = 1
    A[3, 2, 4] = 1
    A[3, 3, 1] = 1
    A[3, 1, 3] = 1
    A[4, 4, 4] = 1
    A[4, 2, 2] = 1
    A[4, 3, 3] = 1
    A[4, 1, 1] = 1

    C = zeros(4, 4)
    C[:, 1] = [0, 0, 0, 0]
    C[:, 2] = [0, 0, 0, pi]
    C[:, 3] = [pi/2, pi/2, pi/2, pi]
    C[:, 4] = [pi/2, pi/2, pi/2, 0]
    return GirvinCorrectionNetworkLearnable(A, C, [false, false])
end

function GirvinCorrectionNetworkLearnable(dv::Integer, dh::Integer; d_out=dh)
    d = Haar(1)
    
    A = Array{Float64, 3}(undef, dv, dh, dh)
    for i in 1:dv
        A[i, :, :] = rand(d, dh)
    end
    C = randn(d_out, dh)
    return GirvinCorrectionNetworkLearnable(A, C, [true, true])
end

function (self::GirvinCorrectionNetworkLearnable)(M::Vector{T}) where T <: Real
    M = M[2:end-1]

    @assert mod(length(M), 2) == 0 "M must have even length (got $(length(M)))"
    M = SpinTokenizer(2)(M)
    angles = nothing
    state = Zygote.@ignore begin
        state = zeros(size(self.C, 1))
        state[1] = 1.
        state
    end

    for i in 1:size(M, 1)
        if angles === nothing
            angles = self.C * state
        else
            angles = hcat(angles, self.C * state)
        end
        angles = hcat(angles, self.C * state)

        v = M[i, :]
        @tensor state[a] := v[j] * state[k] * self.A[j, k, a]
    end
        
    angles = hcat(angles, self.C * state)
    angles = hcat(angles, self.C * state)
    
    return angles'
end

function (self::GirvinCorrectionNetworkLearnable)(M::Matrix{T}) where T <: Real
    bdim = size(M, 2)
    L = size(M, 1)
    M = M[2:end-1, :]
    M = SpinTokenizer(2)(M)
    
    angles = nothing
    state = Zygote.@ignore begin
        state = zeros(size(self.C, 1), bdim)
        state[1, :] .= 1.
        state
    end
    @tensor As[h1, h2, l, b] := self.A[d, h1, h2] * M[l, d, b] 
    
    for i in 1:size(M, 1)
        if angles === nothing
            angles = self.C * state
        else
            angles = hcat(angles, state)
        end
        angles = hcat(angles, state)
        A = As[:, :, i, :]
        state = NNlib.batched_mul(A, reshape(state, size(state, 1), 1, size(state, 2)))[:, 1, :]
    end
        
    angles = hcat(angles, state)
    angles = hcat(angles, state)
    
    angles = self.C * angles
    angles = reshape(angles, :, bdim, L)
    return permutedims(angles, (3, 1, 2))
end


function GirvinMCFeedback(N_state::Int, ancilla_indices::Vector{<:Integer})
    vmodels = [GirvinCircuitIdeal(N_state), GirvinCorrCircuit(Int(N_state/2))]
    dense(x, y) = GirvinCorrectionNetwork()
    model = VariationalMeasurementMCFeedback(vmodels, [dense], ancilla_indices)
    return model
end


# Variational girvin Correction circuit

# Variational circuit with a parametrized two qubit gate
struct VariationalCircuitTwoQubitGateGirvinCorr <: AbstractVariationalCircuit
    params::Array{Float64, 3}
    gate_type::String
    state_indices::Union{Vector{<:Integer}, Nothing}
    VariationalCircuitTwoQubitGateGirvinCorr(params::Array{Float64, 3}; gate_type="CX_Id", state_indices=nothing) = new(params, gate_type, state_indices)
    
end
VariationalCircuitTwoQubitGateGirvinCorr(N::Int, depth::Int; kwargs...) = VariationalCircuitTwoQubitGateGirvinCorr(2π .* rand(N, 2, depth); kwargs...)
VariationalCircuitTwoQubitGateGirvinCorr(; kwargs...) = VariationalCircuitTwoQubitGateGirvinCorr(Array{Float64, 3}(undef, 0, 0, 0); kwargs...) # Empty circuit to be used as a placeholder

Flux.@functor VariationalCircuitTwoQubitGateGirvinCorr
Base.size(model::VariationalCircuitTwoQubitGateGirvinCorr) = size(model.params)
Base.size(model::VariationalCircuitTwoQubitGateGirvinCorr, i::Int) = size(model.params, i)
Circuits.get_depth(model::VariationalCircuitTwoQubitGateGirvinCorr) = size(model.params, 3)
Circuits.get_N(model::VariationalCircuitTwoQubitGateGirvinCorr) = size(model.params, 1) * 3 ÷ 2
Flux.trainable(a::VariationalCircuitTwoQubitGateGirvinCorr) = (a.params,)

function generate_circuit!(circuit, c::VariationalCircuitTwoQubitGateGirvinCorr; params=nothing, N::Integer, depth::Integer)
    for d in 1:depth
        circuit = vcat(circuit, Rylayer(params[:, 1, d]))
        circuit = vcat(circuit, BrickLayer(N, d, params[:, 2, d]; gate=c.gate_type))
    end
    return circuit
end


function Circuits.generate_circuit(c::VariationalCircuitTwoQubitGateGirvinCorr; params=nothing)
    if params === nothing
        params = c.params
    end
    @assert size(params, 2) == 2
    N = size(params, 1)
    depth = size(params, 3)
    N_state = N

    local state_indices
    if c.state_indices === nothing
        state_indices, = Zygote.@ignore get_ancilla_indices(N_state, [false, true, true, true, true, false])
    else
        state_indices = c.state_indices
    end
    circuit = Vector()
    
    for d in 1:depth
        circuit = vcat(circuit, Layers.Rylayer(params[:, 1, d]; sites=state_indices))
        circuit = vcat(circuit, Layers.BrickLayer(N, 1, params[:, 2, d]; gate=c.gate_type, sites=state_indices))
    end
    return circuit
end


struct VariationalCircuitTwoQubitGateGirvinCorrReducedParams <: AbstractVariationalCircuit
    params::Array{Float64, 3}
    gate_type::String
    state_indices::Union{Vector{<:Integer}, Nothing}
    function VariationalCircuitTwoQubitGateGirvinCorrReducedParams(params::Array{Float64, 3}; gate_type="CX_Id", state_indices=nothing)
        @warn "This type is deprecated use VariationalCircuitTwoQubitGateGirvinCorrReducedParamsCorr instead."
        new(params, gate_type, state_indices)
    end
end
VariationalCircuitTwoQubitGateGirvinCorrReducedParams(N::Int, depth::Int; kwargs...) = VariationalCircuitTwoQubitGateGirvinCorrReducedParams(2π .* rand(N ÷ 2, 3, depth) .- π; kwargs...)
VariationalCircuitTwoQubitGateGirvinCorrReducedParams(; kwargs...) = VariationalCircuitTwoQubitGateGirvinCorrReducedParams(Array{Float64, 3}(undef, 0, 0, 0); kwargs...) # Empty circuit to be used as a placeholder

Flux.@functor VariationalCircuitTwoQubitGateGirvinCorrReducedParams
Base.size(model::VariationalCircuitTwoQubitGateGirvinCorrReducedParams) = size(model.params)
Base.size(model::VariationalCircuitTwoQubitGateGirvinCorrReducedParams, i::Int) = size(model.params, i)
Circuits.get_depth(model::VariationalCircuitTwoQubitGateGirvinCorrReducedParams) = size(model.params, 3)
Circuits.get_N(model::VariationalCircuitTwoQubitGateGirvinCorrReducedParams) = size(model.params, 1) * 2
Flux.trainable(a::VariationalCircuitTwoQubitGateGirvinCorrReducedParams) = (a.params,)


function Circuits.generate_circuit(c::VariationalCircuitTwoQubitGateGirvinCorrReducedParams; params=nothing)
    if params === nothing
        params = c.params
    end
    @assert size(params, 2) == 3
    N = size(params, 1) * 2
    depth = size(params, 3)

    local state_indices
    if c.state_indices === nothing
        state_indices, = Zygote.@ignore get_ancilla_indices(N, [false, true, true, true, true, false])
    else
        state_indices = c.state_indices
        @assert length(state_indices) == N
    end
    
    circuit = Vector()
    for d in 1:depth
        params_ry_layer = params[:, 1:2, d]

        circuit = vcat(circuit, Layers.Rylayer(params_ry_layer[:]; sites=state_indices))
        circuit = vcat(circuit, Layers.BrickLayer(N, 1, params[:, 3, d]; gate=c.gate_type, sites=state_indices))
    end
    return circuit
end

"Same as `VariationalCircuitTwoQubitGateGirvinCorrReducedParams` but with the right parameter order"
struct VariationalCircuitTwoQubitGateGirvinCorrReducedParamsCorr <: AbstractVariationalCircuit
    params::Array{Float64, 3}
    gate_type::String
    state_indices::Union{Vector{<:Integer}, Nothing}
    VariationalCircuitTwoQubitGateGirvinCorrReducedParamsCorr(params::Array{Float64, 3}; gate_type="CX_Id", state_indices=nothing) = new(params, gate_type, state_indices)
end
VariationalCircuitTwoQubitGateGirvinCorrReducedParamsCorr(N::Int, depth::Int; kwargs...) = VariationalCircuitTwoQubitGateGirvinCorrReducedParamsCorr(2π .* rand(N ÷ 2, 3, depth) .- π; kwargs...)
VariationalCircuitTwoQubitGateGirvinCorrReducedParamsCorr(; kwargs...) = VariationalCircuitTwoQubitGateGirvinCorrReducedParamsCorr(Array{Float64, 3}(undef, 0, 0, 0); kwargs...) # Empty circuit to be used as a placeholder

Flux.@functor VariationalCircuitTwoQubitGateGirvinCorrReducedParamsCorr
Base.size(model::VariationalCircuitTwoQubitGateGirvinCorrReducedParamsCorr) = size(model.params)
Base.size(model::VariationalCircuitTwoQubitGateGirvinCorrReducedParamsCorr, i::Int) = size(model.params, i)
Circuits.get_depth(model::VariationalCircuitTwoQubitGateGirvinCorrReducedParamsCorr) = size(model.params, 3)
Circuits.get_N(model::VariationalCircuitTwoQubitGateGirvinCorrReducedParamsCorr) = size(model.params, 1) * 2
Flux.trainable(a::VariationalCircuitTwoQubitGateGirvinCorrReducedParamsCorr) = (a.params,)


function Circuits.generate_circuit(c::VariationalCircuitTwoQubitGateGirvinCorrReducedParamsCorr; params=nothing)
    if params === nothing
        params = c.params
    end
    @assert size(params, 2) == 3
    N = size(params, 1) * 2
    depth = size(params, 3)

    local state_indices
    if c.state_indices === nothing
        state_indices, = Zygote.@ignore get_ancilla_indices(N, [false, true, true, true, true, false])
    else
        state_indices = c.state_indices
        @assert length(state_indices) == N
    end
    
    circuit = Vector()
    for d in 1:depth
        params_ry_layer = transpose(params[:, 1:2, d])[:]

        circuit = vcat(circuit, Layers.Rylayer(params_ry_layer; sites=state_indices))
        circuit = vcat(circuit, Layers.BrickLayer(N, 1, params[:, 3, d]; gate=c.gate_type, sites=state_indices))
    end
    return circuit
end


end # module