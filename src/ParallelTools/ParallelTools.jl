# Moved code from here to the package `ParallelGradeint.jl`
# Reimporting ParallelGradient.jl for backcompatibility

using ParallelGradient: dpmap as pmap_diff, dpmap_scalar as pmap_diff_scalar
import Base.+
+(::NamedTuple{(:contents,), Tuple{MPS}}, ::Base.RefValue{Any}) = nothing