module Layers

OneGateLayer(θ; offset=0, gate="Ry", sites=1:length(θ)) = [(gate, sites[i] + offset, (θ = θi,)) for (i, θi) in enumerate(θ)]

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
function CXlayer(N, Π; offset=0, reverse=false)
    start = isodd(Π) ? 1 : 2
    start += offset
    if reverse
        return [("CX", (j+1, j)) for j in start:2:(N - 1)]
    end
    return [("CX", (j, j + 1)) for j in start:2:(N - 1)]
end

# Projective measurement Layer
ProjectiveMeasurementLayer(indices, reset_state) = [("reset", i, (state=reset_state,)) for i in indices]

end