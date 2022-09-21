module Gates

using PastaQ
using ITensors
using ..ITensorsExtension: projective_measurement_gate

function PastaQ.gate(::GateName"reset", st::SiteType"Qubit", s::Index...; state=nothing)
    return projective_measurement_gate(s...; reset=state)
end

PastaQ.is_single_qubit_noise(::GateName"reset") = false

#end
end