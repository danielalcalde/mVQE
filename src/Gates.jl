module Gates

using PastaQ
using ITensors
using ..ITensorsMeasurement: projective_measurement_gate

function PastaQ.gate(::GateName"reset", st::SiteType"Qubit", s::Index...; state=nothing)
    return projective_measurement_gate(s...; reset=state)
end

PastaQ.is_single_qubit_noise(::GateName"reset") = false

function ITensors.op(::OpName"U", ::SiteType"Qubit"; θ::Vector{T}) where T <: Number 
    return [
    exp(1im*(-θ[1]-θ[3])) * cos(θ[2]) -exp(1im*(-θ[1]+θ[3])) * sin(θ[2])
    exp(1im*(θ[1]-θ[3])) * sin(θ[2])  exp(1im*(θ[1]+θ[3])) * cos(θ[2])
    ]
end

function ITensors.op(::OpName"CU", ::SiteType"Qubit"; θ::Vector{T}) where T <: Number
    return [
      1 0 0 0
      0 1 0 0
      0 0 exp(1im*(θ[4]-θ[1]-θ[3])) * cos(θ[2]) -exp(1im*(θ[4]-θ[1]+θ[3])) * sin(θ[2])
      0 0 exp(1im*(θ[4]+θ[1]-θ[3])) * sin(θ[2])  exp(1im*(θ[4]+θ[1]+θ[3])) * cos(θ[2])
    ]
  end

#end
end

function ITensors.op(::OpName"CX_Id", ::SiteType"Qubit"; θ::Number) 
  c = exp(im*θ/2)
  a = 0.5c + 0.5
  b = -0.5c + 0.5
  return [
  1 0 0 0
  0 1 0 0
  0 0 a b
  0 0 b a]
end

function make_U(Hr, Hc)
  H = Hr + im*Hc
  Hs = (H + H') /2
  U = exp(-im * Hs)
  return U
end

function ITensors.op(::OpName"full_U", ::SiteType"Qubit"; H::Array) 
return make_U(H[1, :, :], H[2, :, :])
end