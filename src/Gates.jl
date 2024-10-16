module Gates

using Zygote
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
   Hs = (H + H') ./ 2
   U = exp(im * Hs)
   return U
end

function ITensors.op(::OpName"full_U", ::SiteType"Qubit"; H::Array) 
   return make_U(H[1, :, :], H[2, :, :])
end

function SU_liealgebra(N)
    global SUs
    
    # Check if the SUs has already been defined else create it
    if hasproperty(Gates, :SUs) == false
       SUs = Dict()
    end

    if haskey(SUs, N)
       return SUs[N]
    end

  generators = []
  
  # Hermitian generators
  for i in 1:N
      for j in i:N
          if i == j
              # Diagonal generators
              gen = zeros(ComplexF64, N, N)
              for k in 1:i-1
                  gen[k,k] = 1
              end
              gen[i,i] = 1 - i
              push!(generators, gen)
          else
              # Off-diagonal generators
              gen1 = zeros(ComplexF64, N, N)
              gen1[i,j] = 1
              gen1[j,i] = 1
              push!(generators, gen1)
              
              gen2 = zeros(ComplexF64, N, N)
              gen2[i,j] = -im
              gen2[j,i] = im
              push!(generators, gen2)
          end
      end
  end
  SUs[N] = generators[2:end]
  return generators[2:end]
end

function make_U_lie(θ)
   @assert length(θ) == 15 "θ must be of length 15 and not $(length(θ))"
   g4s = Zygote.@ignore SU_liealgebra(4)
   M = sum(θ .* g4s)
   return exp(im*M)
end

function ITensors.op(::OpName"full_U_lie", ::SiteType"Qubit"; θ::Vector) 
   return make_U_lie(θ)
end