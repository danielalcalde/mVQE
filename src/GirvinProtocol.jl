module GirvinProtocol

using ITensors
using Flux
using Zygote

using ..Circuits: AbstractVariationalCircuit, AbstractVariationalMeasurementCircuit, generate_circuit, VariationalMeasurementMCFeedback
using mVQE: Circuits
using ..Misc: get_ancilla_indices

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
    params = hcat(fill([pi, -pi/2, θ_1, θ_2, pi, θ_1, θ_2, pi], Int(N_state/4))...)'
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

U_gate_L_p(c_in, i1, i2, θ_1, θ_2, ϕ_1) =[("CX", (c_in, i1)), ("CRy", (i1, c_in), (θ = θ_1,)), ("Rx", i1, (θ=ϕ_1,)),("CRy", (i1, i2), (θ = θ_2,)), ("X", i1),
                        ("CX", (c_in, i1)), ("CX", (i1, i2)), ("CX", (i2, i1))]

U_gate_R_p(c_in, i1, i2, θ_1, θ_2, ϕ_1) =[("CX", (c_in, i1)), ("CRy", (i1, c_in), (θ = θ_1,)), ("Rx", i1, (θ=ϕ_1,)),("CRy", (i1, i2), (θ = θ_2,)), ("X", i1),
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
        Mi = M[i, :]
        g = correction_gates_params[Mi]
        for j in 1:2i
            angles[j, :] = add(angles[j, :], g)
        end
    end
    return angles
end
end

function GirvinMCFeedback(N_state::Int, ancilla_indices::Vector{<:Integer})
    vmodels = [GirvinCircuitIdeal(N_state), GirvinCorrCircuit(Int(N_state/2))]
    dense(x, y) = GirvinCorrectionNetwork()
    model = VariationalMeasurementMCFeedback(vmodels, [dense], ancilla_indices)
    return model
end

end # module