using Zygote: @adjoint, _pullback, _tryaxes, last_or_nothing, _tryreverse, _unzip, accum,  _restore
using Functors
import Base.+

# Fix serialization of functions
using Distributed.Serialization: sertag, serialize_type, AbstractSerializer, UNDEFREF_TAG, handle_deserialize, deserialize_cycle
"""
function Distributed.Serialization.serialize(s::AbstractSerializer, x::Core.IntrinsicFunction)
    tag = sertag(x)
    if tag > 0
        return write_as_tag(s.io, tag)
    end
    t = typeof(x)::DataType
    nf = nfields(x)
    if ismutable(x)
        serialize_cycle(s, x) && return
        serialize_type(s, t, true)
    else
        serialize_type(s, t, false)
    end
    for i in 1:nf
        if isdefined(x, i)
            serialize(s, getfield(x, i))
        else
            writetag(s.io, UNDEFREF_TAG)
        end
    end
    nothing
end

function Distributed.Serialization.deserialize(s::AbstractSerializer, t::DataType)
    
    nf = length(t.types)
    if nf == 0 && t.size > 0 && t !== Core.IntrinsicFunction
        # bits type
        return read(s.io, t)
    elseif ismutabletype(t)
        x = ccall(:jl_new_struct_uninit, Any, (Any,), t)
        deserialize_cycle(s, x)
        for i in 1:nf
            tag = Int32(read(s.io, UInt8)::UInt8)
            if tag != UNDEFREF_TAG
                ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), x, i-1, handle_deserialize(s, tag))
            end
        end
        return x
    elseif nf == 0
            yy = ccall(:jl_new_struct_uninit, Any, (Any,), t)
        return yy
    else
        na = nf
        vflds = Vector{Any}(undef, nf)
        for i in 1:nf
            tag = Int32(read(s.io, UInt8)::UInt8)
            if tag != UNDEFREF_TAG
                f = handle_deserialize(s, tag)
                na >= i && (vflds[i] = f)
            else
                na >= i && (na = i - 1) # rest of tail must be undefined values
            end
        end
        return ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), t, vflds, na)
    end
end
"""

_map(f, x...) = map(f, x...)
_map(f, x::Dict, ys...) = Dict(k => f(v, (y[k] for y in ys)...) for (k, v) in x)

+(::NamedTuple{(:contents,), Tuple{MPS}}, ::Base.RefValue{Any}) = nothing
function children_(x::T) where T
    fn = fieldnames(T)
    if length(fn) !=0 && fn[1] isa Symbol
        return NamedTuple((f,getfield(x, f)) for f in fn)
    else
        return Functors.children(x)
    end
end

function update_cache!(cx, Δx, x)
    if cx.cache === nothing
        return nothing
    end
    walk = (recurse, x, ys...) -> _map(recurse, children_(x))
    x = fmap(x->x, x; walk)
    fmap(Δx, x) do Δxi, xi
        if xi in keys(cx.cache)
            if cx.cache[xi] === nothing
                cx.cache[xi] = Δxi
            end
        end
    end
    return cx
end

function pmap_diff(f, iter)
    @distributed_map for i in iter
        f(i)
    end
end

function ∇pmap_diff(cx, f::F, args::Vararg{Any, N}) where {F, N}
    key = Random.randstring(100)
    @everywhere @eval Main begin
        if !isdefined(Main, :pmap_diff_pull_dict)
            pmap_diff_pull_dict = Dict()
        end
        pmap_diff_pull_dict[$key] = Dict()
    end
    
    ys = @distributed_map for args_i = args[1]
        cxi = Zygote.Context()
        y, back = Zygote._pullback(cxi, f, args_i)
        Main.pmap_diff_pull_dict[key][args_i] = back
        y
    end
    arg_ax = map(_tryaxes, args)
    
    function map_back(Δ)
        
        Δf_and_args_zipped = @distributed_map for (args_i, δ) = Tuple(zip(_tryreverse(pmap_diff, args[1], Δ)...))
            Main.pmap_diff_pull_dict[key][args_i](δ)
        end
        @everywhere @eval Main delete!(pmap_diff_pull_dict, $key)
        
        Δf_and_args = _unzip(_tryreverse(pmap_diff, Δf_and_args_zipped), Val(N + 1))
        Δf = reduce(accum, Δf_and_args[1]; init=nothing)
        Δargs = map(_restore, Δf_and_args[2:end], arg_ax)
        
        for s in keys(Δf)
            if isdefined(f, s)
                update_cache!(cx, getfield(Δf, s), getfield(f, s))
            end
        end

        (Δf, Δargs...)
      end
    
    map_back(::Nothing) = nothing
    return ys, map_back
end

@adjoint function pmap_diff(f, args::Union{AbstractArray,Tuple}...)
    ∇pmap_diff(__context__, f, args...)
end


mul_nothing(x, y::Nothing) = nothing
mul_nothing(x::Nothing, y) = nothing
mul_nothing(x, y) = x .* y

pmap_diff_scalar(f, iter) = pmap_diff(f, iter)
function ∇pmap_diff_scalar(cx, f::F, args::Vararg{Any, N}) where {F, N}   
    ys_Δf = @distributed_map for args_i = args[1]
        y, back = Zygote._pullback(cx, f, args_i)
        y, back(1)
    end
    ys = [y for (y, Δf) in ys_Δf]
    Δf_and_args_zipped = [Δf for (y, Δf) in ys_Δf]
    
    arg_ax = map(_tryaxes, args)
    function map_back(Δ)
        # Multiply Δ with Δf_and_args_zipped
        Δf_and_args_zipped = map((Δi, y) -> fmap(z->mul_nothing(z,Δi), y), Δ, Δf_and_args_zipped)
        Δf_and_args = _unzip(_tryreverse(pmap_diff_scalar, Δf_and_args_zipped), Val(N + 1))
        Δf = reduce(accum, Δf_and_args[1]; init=nothing)
        Δargs = map(_restore, Δf_and_args[2:end], arg_ax)
        
        for s in keys(Δf)
            if isdefined(f, s)
                update_cache!(cx, getfield(Δf, s), getfield(f, s))
            end
        end
        
        (Δf, Δargs...)
      end
    
    map_back(::Nothing) = nothing
    return ys, map_back
end

@adjoint function pmap_diff_scalar(f, args::Union{AbstractArray,Tuple}...)
    ∇pmap_diff_scalar(__context__, f, args...)
end

mapsum(func, iterator, args...) = sum(map(iter -> func(iter, args...), iterator))
pmapsum(func, iterator, args...) = sum(pmap_diff(iter -> func(iter, args...), iterator))
pmapsum_scalar(func, iterator, args...) = sum(pmap_diff_scalar(iter -> func(iter, args...), iterator))