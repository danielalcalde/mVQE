module Layers
using Zygote

function OneGateLayer(θ; offset=0, gate="Ry", sites=1:length(θ))
    @assert length(θ) + offset <= length(sites) "Number of parameters must match number of qubits in the layer (param_size=$(length(θ)), qubits=$(length(sites)))"
    return [(gate, sites[i + offset], (θ = θi,)) for (i, θi) in enumerate(θ)]
end

function OneGateLayer2(θ; offset=0, gate="Ry", sites=1:length(θ))
    @assert length(θ) + offset <= length(sites) "Number of parameters must match number of qubits in the layer (param_size=$(length(θ)), qubits=$(length(sites)))"
    a = [(gate, sites[i + offset], (θ = θi,)) for (i, θi) in enumerate(θ)]
    return a
end
Zygote.@adjoint function OneGateLayer2(θ; offset=0, gate="Ry", sites=1:length(θ))
    @assert length(θ) + offset <= length(sites) "Number of parameters must match number of qubits in the layer (param_size=$(length(θ)), qubits=$(length(sites)))"
    a = [(gate, sites[i + offset], (θ = θi,)) for (i, θi) in enumerate(θ)]
    function fg(ā)
        return ([ai[3].θ for ai in ā],)
    end
    return a, fg
end

function print_derivate(x; func=x->x, id="")
    println("Fw_$id: ", func(x)) 
    return x
end
@Zygote.adjoint function print_derivate(x; func=x->x, id="")
    println("Fw_$id: ", func(x))
    function fg(a)
        println("Bw_$id: ", func(a))
        return (a,)
    end
    return x, fg
end

Rzlayer(θ; kwargs...) = OneGateLayer(θ; kwargs..., gate="Rz")
Rylayer(θ; kwargs...) = OneGateLayer(θ; kwargs..., gate="Ry")
Rxlayer(θ; kwargs...) = OneGateLayer(θ; kwargs..., gate="Rx")

Ulayer(θ) = [("U", i, (θ = θ[i, :],)) for i in 1:size(θ, 1)]

# One parameter brick-layer
function BrickLayer(N, Π, θs; offset=0, gate="CX_Id", sites=1:N)
    start = isodd(Π) ? 1 : 2
    start += offset
    return [(gate, (sites[j], sites[j + 1]), (θ = θs[i],)) for (i, j) in enumerate(start:2:(N - 1))]
end

CRxlayer(args...; kwargs...) = BrickLayer(args...; gate="CRx", kwargs...)
CRylayer(args...; kwargs...) = BrickLayer(args...; gate="CRy", kwargs...)
CRzlayer(args...; kwargs...) = BrickLayer(args...; gate="CRz", kwargs...)

CX_Idlayer(args...; kwargs...) = BrickLayer(args...; gate="CX_Id", kwargs...)


function CUlayer(N, Π, θs; offset=0)
    start = isodd(Π) ? 1 : 2
    start += offset
    @assert size(θs, 2) == 4 "θs must be of shape (4, depth) and not $(size(θs))"
    return [("CU", (j, j + 1), (θ = θs[i, :],)) for (i, j) in enumerate(start:2:(N - 1))]
end

function CUlayer_broken(N, Π, θs; broken=6)
    start = isodd(Π) ? 1 : 2
    @assert size(θs, 2) == 4 "θs must be of shape (4, depth) and not $(size(θs))"
    return [("CU", (j, j + 1), (θ = θs[i, :],)) for (i, j) in enumerate(start:2:(N - 1)) if mod(j, broken) != 0]
end

# brick-layer of CX gates
function CXlayer(N, Π; offset=0, reverse=false, periodic=false, sites=1:N, holes=Int[])
return Zygote.ignore() do
    start = isodd(Π) ? 1 : 2
    start += offset
    local f, Nmax
    N = length(sites)
    if periodic
        f = j-> sites[mod1(j, N)]
        Nmax = N
    else
        f = j->sites[j]
        Nmax = N - 1
    end
    
    layer = Tuple[]
    for j in start:2:Nmax
        if reverse
            t = ("CX", (f(j+1), f(j)))
        else
            t = ("CX", (f(j), f(j+1)))
        end
        # If there are holes between j and j+1, skip the gate
        if ! any(f(j) .< holes .< f(j + 1)) 
            push!(layer, t)
        end
    end
    return layer
end
end


# Full 2Body layer
function FullTwoBody(N, Π, θs; offset=0, sites=1:N)
    @assert N <= size(θs, 1)*2 "Number of qubits must be less or equal than the number of parameters"
    start = isodd(Π) ? 1 : 2
    start += offset
    return [("full_U_lie", (sites[j], sites[j + 1]), (θ = θs[i, :],)) for (i, j) in enumerate(start:2:(N - 1))]
end


# Bell gate
BellGate(i0, i1) = [("CX", (i1, i0)), ("H", i1)]
function BellGateLayer(N; sites=1:N)
    @assert N % 2 == 0 "Number of qubits must be even"
    return vcat([BellGate(sites[i], sites[i + 1]) for i in 1:2:length(sites)]...)
end


# Projective measurement Layer
ProjectiveMeasurementLayer(indices, reset_state) = [("reset", i, (state=reset_state,)) for i in indices]



end