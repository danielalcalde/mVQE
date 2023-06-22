module Layers

Rzlayer(θ; offset=0) = [("Rz", i + offset, (θ = θi,)) for (i, θi) in enumerate(θ)]
Rylayer(θ; offset=0) = [("Ry", i + offset, (θ = θi,)) for (i, θi) in enumerate(θ)]
Rxlayer(θ; offset=0) = [("Rx", i + offset, (θ = θi,)) for (i, θi) in enumerate(θ)]
Ulayer(θ) = [("U", i, (θ = θ[i, :],)) for i in 1:size(θ, 1)]

# brick-layer of CRX gates
function CRxlayer(N, Π, θs)
    start = isodd(Π) ? 1 : 2
    return [("CRx", (j, j + 1), (θ = θs[i],)) for (i, j) in enumerate(start:2:(N - 1))]
end

# brick-layer of CRX gates
function CRylayer(N, Π, θs)
    start = isodd(Π) ? 1 : 2
    return [("CRy", (j, j + 1), (θ = θs[i],)) for (i, j) in enumerate(start:2:(N - 1))]
end

function CUlayer(N, Π, θs)
    start = isodd(Π) ? 1 : 2
    @assert size(θs, 2) == 4 "θs must be of shape (4, depth) and not $(size(θs))"
    return [("CU", (j, j + 1), (θ = θs[i, :],)) for (i, j) in enumerate(start:2:(N - 1))]
end

function CUlayer_broken(N, Π, θs; broken=6)
    start = isodd(Π) ? 1 : 2
    @assert size(θs, 2) == 4 "θs must be of shape (4, depth) and not $(size(θs))"
    return [("CU", (j, j + 1), (θ = θs[i, :],)) for (i, j) in enumerate(start:2:(N - 1)) if mod(j, broken) != 0]
end

# brick-layer of CX gates
function CXlayer(N, Π)
    start = isodd(Π) ? 1 : 2
    return [("CX", (j, j + 1)) for j in start:2:(N - 1)]
end

# Projective measurement Layer
ProjectiveMeasurementLayer(indices, reset_state) = [("reset", i, (state=reset_state,)) for i in indices]

end