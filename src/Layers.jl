module Layers

Rzlayer(θ) = [("Rz", i, (θ = θi,)) for (i, θi) in enumerate(θ)]
Rylayer(θ) = [("Ry", i, (θ = θi,)) for (i, θi) in enumerate(θ)]
Rxlayer(θ) = [("Rx", i, (θ = θi,)) for (i, θi) in enumerate(θ)]

# brick-layer of CRX gates
function CRxlayer(N, Π, θs)
    start = isodd(Π) ? 1 : 2
    return [("CRx", (j, j + 1), (θ = θs[i],)) for (i, j) in enumerate(start:2:(N - 1))]
end

# brick-layer of CX gates
function CXlayer(N, Π)
    start = isodd(Π) ? 1 : 2
    return [("CX", (j, j + 1)) for j in start:2:(N - 1)]
end

# Projective measurement Layer
ProjectiveMeasurementLayer(indices, reset_state) = [("reset", i, (state=reset_state,)) for i in indices]

end