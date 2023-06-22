using ITensors
using PastaQ
using mVQE
using Random
using Zygote
using Test

for file in readlines(joinpath(@__DIR__, "testgroups"))
    println("Testing $file.jl")
    include(file * ".jl")
end