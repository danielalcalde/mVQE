using ZygoteRules
using Zygote: ChainRulesCore, ChainRules

precompile(Tuple{mVQE.var"#optimize_and_evolve##kw", NamedTuple{(:optimizer, :verbose, :maxdims, :noise), Tuple{OptimKit.LBFGS{Float64, OptimKit.HagerZhangLineSearch{Base.Rational{Int64}}}, Bool, Int64, Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}}}, typeof(mVQE.optimize_and_evolve), Int64, Array{Int64, 1}, ITensors.MPO, ITensors.MPO, mVQE.Circuits.VariationalCircuitRy, Int64})

precompile(Tuple{mVQE.ITensorsExtension.var"#projective_measurement##kw", NamedTuple{(:indices, :reset), Tuple{Array{Int64, 1}, Int64}}, typeof(mVQE.ITensorsExtension.projective_measurement), ITensors.MPO})

precompile(Tuple{typeof(Zygote.ignore), mVQE.ITensorsExtension.var"#6#7"{ITensors.ITensor, ITensors.Index{Int64}, ITensors.Index{Int64}, ITensors.Index{Int64}, ITensors.Index{Int64}}})

precompile(Tuple{typeof(Base.convert), Type{UnionAll}, Type{Zygote.Pullback{Tuple{PastaQ.var"#runcircuit##kw", NamedTuple{(:noise, :maxdims), Tuple{Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}, Int64}}, typeof(PastaQ.runcircuit), ITensors.MPO, mVQE.Circuits.VariationalCircuitRy, Array{Array{Float64, 1}, 1}}, T} where T}})

precompile(Tuple{typeof(Base.convert), Type{UnionAll}, Type{Zygote.Pullback{Tuple{Type{Base.Generator{I, F} where F where I}, mVQE.Layers.var"#5#6", Base.StepRange{Int64, Int64}}, T} where T}})

precompile(Tuple{typeof(Base.convert), Type{UnionAll}, Type{Zygote.Pullback{Tuple{Type{Base.Generator{Base.StepRange{Int64, Int64}, mVQE.Layers.var"#5#6"}}, mVQE.Layers.var"#5#6", Base.StepRange{Int64, Int64}}, T} where T}})

precompile(Tuple{typeof(Base.convert), Type{UnionAll}, Type{Zygote.Pullback{Tuple{typeof(Base.convert), Type{mVQE.Layers.var"#5#6"}, mVQE.Layers.var"#5#6"}, T} where T}})

precompile(Tuple{typeof(Base.convert), Type{UnionAll}, Type{Zygote.Pullback{Tuple{mVQE.Layers.var"#5#6", Int64}, T} where T}})

precompile(Tuple{typeof(Base.convert), Type{UnionAll}, Type{Zygote.Pullback{Tuple{typeof(mVQE.Layers.Rylayer), Float64}, T} where T}})

precompile(Tuple{typeof(Base.convert), Type{UnionAll}, Type{Zygote.Pullback{Tuple{Type{Base.Generator{I, F} where F where I}, mVQE.Layers.var"#1#2", Base.Iterators.Enumerate{Float64}}, T} where T}})

precompile(Tuple{typeof(Base.convert), Type{UnionAll}, Type{Zygote.Pullback{Tuple{Type{Base.Generator{Base.Iterators.Enumerate{Float64}, mVQE.Layers.var"#1#2"}}, mVQE.Layers.var"#1#2", Base.Iterators.Enumerate{Float64}}, T} where T}})

precompile(Tuple{typeof(Base.convert), Type{UnionAll}, Type{Zygote.Pullback{Tuple{typeof(Base.convert), Type{mVQE.Layers.var"#1#2"}, mVQE.Layers.var"#1#2"}, T} where T}})

precompile(Tuple{typeof(Base.convert), Type{UnionAll}, Type{Zygote.Pullback{Tuple{mVQE.Layers.var"#1#2", Tuple{Int64, Float64}}, T} where T}})

precompile(Tuple{typeof(Base.convert), Type{UnionAll}, Type{Zygote.Pullback{Tuple{typeof(mVQE.Layers.Rylayer), Array{Float64, 1}}, T} where T}})

precompile(Tuple{typeof(Base.convert), Type{UnionAll}, Type{Zygote.Pullback{Tuple{Type{Base.Generator{I, F} where F where I}, mVQE.Layers.var"#1#2", Base.Iterators.Enumerate{Array{Float64, 1}}}, T} where T}})

precompile(Tuple{typeof(Base.convert), Type{UnionAll}, Type{Zygote.Pullback{Tuple{Type{Base.Generator{Base.Iterators.Enumerate{Array{Float64, 1}}, mVQE.Layers.var"#1#2"}}, mVQE.Layers.var"#1#2", Base.Iterators.Enumerate{Array{Float64, 1}}}, T} where T}})

precompile(Tuple{typeof(Base.convert), Type{UnionAll}, Type{Zygote.Pullback{Tuple{Type{mVQE.Circuits.Circuit}, Array{Tuple, 1}}, T} where T}})

precompile(Tuple{typeof(Base.convert), Type{UnionAll}, Type{Zygote.Pullback{Tuple{PastaQ.var"#runcircuit##kw", NamedTuple{(:noise, :maxdims), Tuple{Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}, Int64}}, typeof(PastaQ.runcircuit), ITensors.MPO, mVQE.Circuits.Circuit}, T} where T}})

precompile(Tuple{mVQE.var"#optimize_and_evolve##kw", NamedTuple{(:θ, :optimizer, :maxdims, :noise), Tuple{Nothing, OptimKit.LBFGS{Float64, OptimKit.HagerZhangLineSearch{Base.Rational{Int64}}}, Int64, Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}}}, typeof(mVQE.optimize_and_evolve), ITensors.MPO, ITensors.MPO, mVQE.Circuits.VariationalCircuitRy, Int64})

precompile(Tuple{OptimKit.var"##optimize#11", typeof(OptimKit._precondition), typeof(OptimKit._finalize!), Function, typeof(OptimKit._inner), typeof(OptimKit._transport!), typeof(OptimKit._scale!), typeof(OptimKit._add!), Bool, typeof(OptimKit.optimize), mVQE.var"#loss_and_grad#7"{mVQE.var"#loss_#6"{Base.Pairs{Symbol, Any, Tuple{Symbol, Symbol}, NamedTuple{(:maxdims, :noise), Tuple{Int64, Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}}}}, ITensors.MPO, ITensors.MPO, mVQE.Circuits.VariationalCircuitRy}}, Array{Array{Float64, 1}, 1}, OptimKit.LBFGS{Float64, OptimKit.HagerZhangLineSearch{Base.Rational{Int64}}}})

precompile(Tuple{typeof(ZygoteRules._pullback), Zygote.Context, mVQE.var"#loss_#6"{Base.Pairs{Symbol, Any, Tuple{Symbol, Symbol}, NamedTuple{(:maxdims, :noise), Tuple{Int64, Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}}}}, ITensors.MPO, ITensors.MPO, mVQE.Circuits.VariationalCircuitRy}, Array{Array{Float64, 1}, 1}})

precompile(Tuple{typeof(Base.convert), Type{UnionAll}, Type{Zygote.Pullback{Tuple{mVQE.var"#loss##kw", NamedTuple{(:maxdims,), Tuple{Int64}}, typeof(mVQE.loss), ITensors.MPO, ITensors.MPO}, T} where T}})

precompile(Tuple{typeof(ZygoteRules._pullback), Zygote.Context, mVQE.var"#loss##kw", NamedTuple{(:maxdims,), Tuple{Int64}}, typeof(mVQE.loss), ITensors.MPO, ITensors.MPO})

precompile(Tuple{typeof(Base.indexed_iterate), Tuple{Float64, Zygote.Pullback{Tuple{mVQE.var"#loss_#6"{Base.Pairs{Symbol, Any, Tuple{Symbol, Symbol}, NamedTuple{(:maxdims, :noise), Tuple{Int64, Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}}}}, ITensors.MPO, ITensors.MPO, mVQE.Circuits.VariationalCircuitRy}, Array{Array{Float64, 1}, 1}}, Any}}, Int64})

precompile(Tuple{typeof(Base.indexed_iterate), Tuple{Float64, Zygote.Pullback{Tuple{mVQE.var"#loss_#6"{Base.Pairs{Symbol, Any, Tuple{Symbol, Symbol}, NamedTuple{(:maxdims, :noise), Tuple{Int64, Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}}}}, ITensors.MPO, ITensors.MPO, mVQE.Circuits.VariationalCircuitRy}, Array{Array{Float64, 1}, 1}}, Any}}, Int64, Int64})

precompile(Tuple{Zygote.var"#52#53"{Zygote.Pullback{Tuple{mVQE.var"#loss_#6"{Base.Pairs{Symbol, Any, Tuple{Symbol, Symbol}, NamedTuple{(:maxdims, :noise), Tuple{Int64, Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}}}}, ITensors.MPO, ITensors.MPO, mVQE.Circuits.VariationalCircuitRy}, Array{Array{Float64, 1}, 1}}, Any}}, Float64})

precompile(Tuple{Zygote.Pullback{Tuple{mVQE.var"#loss##kw", NamedTuple{(:maxdims, :noise), Tuple{Int64, Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}}}, typeof(mVQE.loss), ITensors.MPO, ITensors.MPO, mVQE.Circuits.VariationalCircuitRy, Array{Array{Float64, 1}, 1}}, Any}, Float64})

precompile(Tuple{Zygote.Pullback{Tuple{mVQE.var"##loss#3", Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}, Base.Pairs{Symbol, Int64, Tuple{Symbol}, NamedTuple{(:maxdims,), Tuple{Int64}}}, typeof(mVQE.loss), ITensors.MPO, ITensors.MPO, mVQE.Circuits.VariationalCircuitRy, Array{Array{Float64, 1}, 1}}, Any}, Float64})

precompile(Tuple{Zygote.Pullback{Tuple{mVQE.var"#loss##kw", NamedTuple{(:maxdims,), Tuple{Int64}}, typeof(mVQE.loss), ITensors.MPO, ITensors.MPO}, Tuple{Zygote.Pullback{Tuple{mVQE.var"##loss#2", Base.Pairs{Symbol, Int64, Tuple{Symbol}, NamedTuple{(:maxdims,), Tuple{Int64}}}, typeof(mVQE.loss), ITensors.MPO, ITensors.MPO}, Any}, Zygote.var"#2093#back#357"{Zygote.var"#pairs_namedtuple_pullback#356"{(:maxdims,), NamedTuple{(:maxdims,), Tuple{Int64}}}}}}, Float64})

precompile(Tuple{Zygote.Pullback{Tuple{PastaQ.var"#runcircuit##kw", NamedTuple{(:noise, :maxdims), Tuple{Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}, Int64}}, typeof(PastaQ.runcircuit), ITensors.MPO, mVQE.Circuits.VariationalCircuitRy, Array{Array{Float64, 1}, 1}}, Tuple{Zygote.Pullback{Tuple{mVQE.Circuits.var"##runcircuit#9", Base.Pairs{Symbol, Any, Tuple{Symbol, Symbol}, NamedTuple{(:noise, :maxdims), Tuple{Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}, Int64}}}, typeof(PastaQ.runcircuit), ITensors.MPO, mVQE.Circuits.VariationalCircuitRy, Array{Array{Float64, 1}, 1}}, Any}, Zygote.var"#2093#back#357"{Zygote.var"#pairs_namedtuple_pullback#356"{(:noise, :maxdims), NamedTuple{(:noise, :maxdims), Tuple{Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}, Int64}}}}}}, ITensors.MPO})

precompile(Tuple{Zygote.Pullback{Tuple{PastaQ.var"#runcircuit##kw", NamedTuple{(:noise, :maxdims), Tuple{Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}, Int64}}, typeof(PastaQ.runcircuit), ITensors.MPO, mVQE.Circuits.Circuit}, Tuple{Zygote.Pullback{Tuple{mVQE.Circuits.var"##runcircuit#13", Base.Pairs{Symbol, Any, Tuple{Symbol, Symbol}, NamedTuple{(:noise, :maxdims), Tuple{Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}, Int64}}}, typeof(PastaQ.runcircuit), ITensors.MPO, mVQE.Circuits.Circuit}, Any}, Zygote.var"#2093#back#357"{Zygote.var"#pairs_namedtuple_pullback#356"{(:noise, :maxdims), NamedTuple{(:noise, :maxdims), Tuple{Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}, Int64}}}}}}, ITensors.MPO})

precompile(Tuple{Zygote.var"#1761#back#218"{Zygote.var"#back#217"{:circ, Zygote.Context, mVQE.Circuits.Circuit, Array{Tuple, 1}}}, Array{Union{Nothing, Tuple{Nothing, Nothing, NamedTuple{(:θ,), Tuple{Float64}}}}, 1}})

precompile(Tuple{Type{ChainRulesCore.Tangent{mVQE.Circuits.Circuit, NamedTuple{(:circ,), Tuple{Array{ChainRulesCore.AbstractTangent, 1}}}}}, NamedTuple{(:circ,), Tuple{Array{ChainRulesCore.AbstractTangent, 1}}}})

precompile(Tuple{typeof(ChainRulesCore.canonicalize), ChainRulesCore.Tangent{mVQE.Circuits.Circuit, NamedTuple{(:circ,), Tuple{Array{ChainRulesCore.AbstractTangent, 1}}}}})

precompile(Tuple{typeof(Zygote.wrap_chainrules_output), ChainRulesCore.Tangent{mVQE.Circuits.Circuit, NamedTuple{(:circ,), Tuple{Array{ChainRulesCore.AbstractTangent, 1}}}}})

precompile(Tuple{typeof(Base.map), Function, ChainRulesCore.Tangent{mVQE.Circuits.Circuit, NamedTuple{(:circ,), Tuple{Array{ChainRulesCore.AbstractTangent, 1}}}}})

precompile(Tuple{Type{ChainRulesCore.Tangent{mVQE.Circuits.Circuit, NamedTuple{(:circ,), Tuple{Array{Union{Nothing, Tuple{Nothing, Nothing, NamedTuple{(:θ,), Tuple{Float64}}}}, 1}}}}}, NamedTuple{(:circ,), Tuple{Array{Union{Nothing, Tuple{Nothing, Nothing, NamedTuple{(:θ,), Tuple{Float64}}}}, 1}}}})

precompile(Tuple{Zygote.Pullback{Tuple{typeof(mVQE.Circuits.generate_circuit), mVQE.Circuits.VariationalCircuitRy, Array{Array{Float64, 1}, 1}}, Any}, NamedTuple{(:circ,), Tuple{Array{Union{Nothing, Tuple{Nothing, Nothing, NamedTuple{(:θ,), Tuple{Float64}}}}, 1}}}})

precompile(Tuple{Zygote.Pullback{Tuple{Type{mVQE.Circuits.Circuit}, Array{Tuple, 1}}, Tuple{Zygote.var"#1784#back#224"{Zygote.Jnew{mVQE.Circuits.Circuit, Nothing, false}}}}, NamedTuple{(:circ,), Tuple{Array{Union{Nothing, Tuple{Nothing, Nothing, NamedTuple{(:θ,), Tuple{Float64}}}}, 1}}}})

precompile(Tuple{Zygote.Pullback{Tuple{typeof(mVQE.Layers.Rylayer), Array{Float64, 1}}, Tuple{Zygote.Pullback{Tuple{Type{Base.Generator{I, F} where F where I}, mVQE.Layers.var"#1#2", Base.Iterators.Enumerate{Array{Float64, 1}}}, Tuple{Zygote.Pullback{Tuple{Type{Base.Generator{Base.Iterators.Enumerate{Array{Float64, 1}}, mVQE.Layers.var"#1#2"}}, mVQE.Layers.var"#1#2", Base.Iterators.Enumerate{Array{Float64, 1}}}, Tuple{Zygote.Pullback{Tuple{typeof(Base.convert), Type{mVQE.Layers.var"#1#2"}, mVQE.Layers.var"#1#2"}, Tuple{}}, Zygote.var"#1784#back#224"{Zygote.Jnew{Base.Generator{Base.Iterators.Enumerate{Array{Float64, 1}}, mVQE.Layers.var"#1#2"}, Nothing, false}}, Zygote.Pullback{Tuple{typeof(Base.convert), Type{Base.Iterators.Enumerate{Array{Float64, 1}}}, Base.Iterators.Enumerate{Array{Float64, 1}}}, Tuple{}}}}}}, Zygote.var"#1784#back#224"{Zygote.Jnew{mVQE.Layers.var"#1#2", Nothing, false}}, Zygote.var"#2656#back#593"{Zygote.var"#back#592"}, Zygote.var"#back#578"{Zygote.var"#map_back#548"{mVQE.Layers.var"#1#2", 1, Tuple{Base.Iterators.Enumerate{Array{Float64, 1}}}, Tuple{Tuple{Base.OneTo{Int64}}}, Array{Tuple{Tuple{String, Int64, NamedTuple{(:θ,), Tuple{Float64}}}, Zygote.Pullback{Tuple{mVQE.Layers.var"#1#2", Tuple{Int64, Float64}}, Tuple{Zygote.Pullback{Tuple{Type{NamedTuple{(:θ,), T} where T<:Tuple}, Tuple{Float64}}, Tuple{Zygote.Pullback{Tuple{Type{NamedTuple{(:θ,), Tuple{Float64}}}, Tuple{Float64}}, Tuple{Zygote.var"#1794#back#226"{Zygote.Jnew{NamedTuple{(:θ,), Tuple{Float64}}, Nothing, true}}}}}}, Zygote.var"#1642#back#160"{Zygote.var"#back#158"{2, 1, Zygote.Context, Float64}}, Zygote.var"#1642#back#160"{Zygote.var"#back#158"{2, 2, Zygote.Context, Int64}}, Zygote.var"#1630#back#155"{typeof(Base.identity)}, Zygote.var"#back#178"{Zygote.var"#1642#back#160"{Zygote.var"#back#158"{2, 1, Zygote.Context, Int64}}}, Zygote.var"#1630#back#155"{typeof(Base.identity)}, Zygote.var"#1642#back#160"{Zygote.var"#back#158"{2, 1, Zygote.Context, Int64}}, Zygote.var"#back#179"{Zygote.var"#1642#back#160"{Zygote.var"#back#158"{2, 2, Zygote.Context, Float64}}}}}}, 1}}}}}, Array{Union{Nothing, Tuple{Nothing, Nothing, NamedTuple{(:θ,), Tuple{Float64}}}}, 1}})

precompile(Tuple{Zygote.Pullback{Tuple{Type{Base.Generator{I, F} where F where I}, mVQE.Layers.var"#1#2", Base.Iterators.Enumerate{Array{Float64, 1}}}, Tuple{Zygote.Pullback{Tuple{Type{Base.Generator{Base.Iterators.Enumerate{Array{Float64, 1}}, mVQE.Layers.var"#1#2"}}, mVQE.Layers.var"#1#2", Base.Iterators.Enumerate{Array{Float64, 1}}}, Tuple{Zygote.Pullback{Tuple{typeof(Base.convert), Type{mVQE.Layers.var"#1#2"}, mVQE.Layers.var"#1#2"}, Tuple{}}, Zygote.var"#1784#back#224"{Zygote.Jnew{Base.Generator{Base.Iterators.Enumerate{Array{Float64, 1}}, mVQE.Layers.var"#1#2"}, Nothing, false}}, Zygote.Pullback{Tuple{typeof(Base.convert), Type{Base.Iterators.Enumerate{Array{Float64, 1}}}, Base.Iterators.Enumerate{Array{Float64, 1}}}, Tuple{}}}}}}, NamedTuple{(:f, :iter), Tuple{Nothing, Array{Tuple{Nothing, Float64}, 1}}}})

precompile(Tuple{Zygote.Pullback{Tuple{typeof(mVQE.Layers.CXlayer), Int64, Int64}, Any}, Array{Union{Nothing, Tuple{Nothing, Nothing, NamedTuple{(:θ,), Tuple{Float64}}}}, 1}})

precompile(Tuple{Zygote.var"#back#578"{Zygote.var"#map_back#548"{mVQE.Layers.var"#5#6", 1, Tuple{Base.StepRange{Int64, Int64}}, Tuple{Tuple{Base.OneTo{Int64}}}, Array{Tuple{Tuple{String, Tuple{Int64, Int64}}, Zygote.Pullback{Tuple{mVQE.Layers.var"#5#6", Int64}, Tuple{Zygote.var"#1630#back#155"{typeof(Base.identity)}, Zygote.var"#1870#back#270"{Zygote.var"#266#268"{Tuple{Int64, Int64}}}, Zygote.var"#1630#back#155"{typeof(Base.identity)}}}}, 1}}}, Array{Union{Nothing, Tuple{Nothing, Nothing, NamedTuple{(:θ,), Tuple{Float64}}}}, 1}})

precompile(Tuple{Zygote.Pullback{Tuple{Type{Base.Generator{I, F} where F where I}, mVQE.Layers.var"#5#6", Base.StepRange{Int64, Int64}}, Tuple{Zygote.Pullback{Tuple{Type{Base.Generator{Base.StepRange{Int64, Int64}, mVQE.Layers.var"#5#6"}}, mVQE.Layers.var"#5#6", Base.StepRange{Int64, Int64}}, Tuple{Zygote.Pullback{Tuple{typeof(Base.convert), Type{mVQE.Layers.var"#5#6"}, mVQE.Layers.var"#5#6"}, Tuple{}}, Zygote.var"#1784#back#224"{Zygote.Jnew{Base.Generator{Base.StepRange{Int64, Int64}, mVQE.Layers.var"#5#6"}, Nothing, false}}, Zygote.Pullback{Tuple{typeof(Base.convert), Type{Base.StepRange{Int64, Int64}}, Base.StepRange{Int64, Int64}}, Any}}}}}, NamedTuple{(:f, :iter), Tuple{Nothing, Array{Nothing, 1}}}})

precompile(Tuple{Zygote.var"#1761#back#218"{Zygote.var"#back#217"{:H, Zygote.Context, mVQE.var"#loss_#6"{Base.Pairs{Symbol, Any, Tuple{Symbol, Symbol}, NamedTuple{(:maxdims, :noise), Tuple{Int64, Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}}}}, ITensors.MPO, ITensors.MPO, mVQE.Circuits.VariationalCircuitRy}, ITensors.MPO}}, ITensors.MPO})

precompile(Tuple{Zygote.var"#1761#back#218"{Zygote.var"#back#217"{:ψs, Zygote.Context, mVQE.var"#loss_#6"{Base.Pairs{Symbol, Any, Tuple{Symbol, Symbol}, NamedTuple{(:maxdims, :noise), Tuple{Int64, Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}}}}, ITensors.MPO, ITensors.MPO, mVQE.Circuits.VariationalCircuitRy}, ITensors.MPO}}, ITensors.MPO})

precompile(Tuple{Type{OptimKit.HagerZhangLineSearchIterator{T₁, F₁, F₂, F₃, X, G, T₂} where T₂<:Real where G where X where F₃ where F₂ where F₁ where T₁<:Real}, mVQE.var"#loss_and_grad#7"{mVQE.var"#loss_#6"{Base.Pairs{Symbol, Any, Tuple{Symbol, Symbol}, NamedTuple{(:maxdims, :noise), Tuple{Int64, Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}}}}, ITensors.MPO, ITensors.MPO, mVQE.Circuits.VariationalCircuitRy}}, typeof(OptimKit._retract), typeof(OptimKit._inner), OptimKit.LineSearchPoint{Float64, Array{Array{Float64, 1}, 1}, Array{Array{Float64, 1}, 1}}, Array{Array{Float64, 1}, 1}, Float64, Bool, OptimKit.HagerZhangLineSearch{Base.Rational{Int64}}})

precompile(Tuple{typeof(Base.iterate), OptimKit.HagerZhangLineSearchIterator{Float64, mVQE.var"#loss_and_grad#7"{mVQE.var"#loss_#6"{Base.Pairs{Symbol, Any, Tuple{Symbol, Symbol}, NamedTuple{(:maxdims, :noise), Tuple{Int64, Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}}}}, ITensors.MPO, ITensors.MPO, mVQE.Circuits.VariationalCircuitRy}}, typeof(OptimKit._retract), typeof(OptimKit._inner), Array{Array{Float64, 1}, 1}, Array{Array{Float64, 1}, 1}, Base.Rational{Int64}}})

precompile(Tuple{mVQE.var"#loss_and_grad#7"{mVQE.var"#loss_#6"{Base.Pairs{Symbol, Any, Tuple{Symbol, Symbol}, NamedTuple{(:maxdims, :noise), Tuple{Int64, Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}}}}, ITensors.MPO, ITensors.MPO, mVQE.Circuits.VariationalCircuitRy}}, Array{Array{Float64, 1}, 1}})

precompile(Tuple{typeof(OptimKit.bracket), OptimKit.HagerZhangLineSearchIterator{Float64, mVQE.var"#loss_and_grad#7"{mVQE.var"#loss_#6"{Base.Pairs{Symbol, Any, Tuple{Symbol, Symbol}, NamedTuple{(:maxdims, :noise), Tuple{Int64, Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}}}}, ITensors.MPO, ITensors.MPO, mVQE.Circuits.VariationalCircuitRy}}, typeof(OptimKit._retract), typeof(OptimKit._inner), Array{Array{Float64, 1}, 1}, Array{Array{Float64, 1}, 1}, Base.Rational{Int64}}, OptimKit.LineSearchPoint{Float64, Array{Array{Float64, 1}, 1}, Array{Array{Float64, 1}, 1}}})

precompile(Tuple{typeof(OptimKit.bisect), OptimKit.HagerZhangLineSearchIterator{Float64, mVQE.var"#loss_and_grad#7"{mVQE.var"#loss_#6"{Base.Pairs{Symbol, Any, Tuple{Symbol, Symbol}, NamedTuple{(:maxdims, :noise), Tuple{Int64, Tuple{Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}, Pair{Int64, Tuple{String, NamedTuple{(:p,), Tuple{Float64}}}}}}}}, ITensors.MPO, ITensors.MPO, mVQE.Circuits.VariationalCircuitRy}}, typeof(OptimKit._retract), typeof(OptimKit._inner), Array{Array{Float64, 1}, 1}, Array{Array{Float64, 1}, 1}, Base.Rational{Int64}}, OptimKit.LineSearchPoint{Float64, Array{Array{Float64, 1}, 1}, Array{Array{Float64, 1}, 1}}, OptimKit.LineSearchPoint{Float64, Array{Array{Float64, 1}, 1}, Array{Array{Float64, 1}, 1}}})
