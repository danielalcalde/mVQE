using Distributed: splitrange

macro distributed_fast(args...)
    na = length(args)
    if na==1
        loop = args[1]
    elseif na==2
        reducer = args[1]
        loop = args[2]
    else
        throw(ArgumentError("wrong number of arguments to @distributed"))
    end
    if !isa(loop,Expr) || loop.head !== :for
        error("malformed @distributed loop")
    end
    var = loop.args[1].args[1]
    r = loop.args[1].args[2]
    body = loop.args[2]
    if Meta.isexpr(body, :block) && body.args[end] isa LineNumberNode
        resize!(body.args, length(body.args) - 1)
    end
    if na==1
        syncvar = esc(Base.sync_varname)
        return quote
            local ref = pfor($(make_pfor_body(var, body)), $(esc(r)))
            if $(Expr(:islocal, syncvar))
                put!($syncvar, ref)
            end
            ref
        end
    else
        #o = :(($(esc(reducer)), $(make_preduce_body(var, body)), $(esc(r))))
        #println(o)              
        return :(preduce2($(esc(reducer)), $(make_preduce_body2(var, body)), $(esc(r))))
    end
end

function make_preduce_body2(var, body)
    quote
        function (reducer, R)
            $(esc(var)) = R[1]
            ac = $(esc(body))
            if length(R) > 1
                for $(esc(var)) in R[2:end]
                    ac = reducer(ac, $(esc(body)))
                end
            end
            ac
        end
    end
end

function preduce2(reducer, f, R)
    chunks = splitrange(Int(firstindex(R)), Int(lastindex(R)), nworkers())
    all_w = workers()[1:length(chunks)]
    
    w_exec = Task[]
    for (idx,pid) in enumerate(all_w)
        b, e = first(chunks[idx]), last(chunks[idx])
        t = Task(()->remotecall_fetch(f, pid, reducer, R[b:e]))
        schedule(t)
        push!(w_exec, t)
    end
    reduce(reducer, Any[fetch(t) for t in w_exec])
end

using Distributed: splitrange

macro distributed_map(loop)
    if !isa(loop,Expr) || loop.head !== :for
        error("malformed @distributed loop")
    end
    var = loop.args[1].args[1]
    r = loop.args[1].args[2]
    body = loop.args[2]
    if Meta.isexpr(body, :block) && body.args[end] isa LineNumberNode
        resize!(body.args, length(body.args) - 1)
    end
       
    return :(preduce_map($(make_preduce_body_map(var, body)), $(esc(r))))

end

function make_preduce_body_map(var, body)
    quote
        function (R)
            accum = Vector{Any}(undef, length(R))
            for (i, $(esc(var))) in enumerate(R)
                accum[i] = $(esc(body))
            end
            accum
        end
    end
end

function preduce_map(f, R)
    chunks = splitrange(Int(firstindex(R)), Int(lastindex(R)), nworkers())
    all_w = workers()[1:length(chunks)]
    
    w_exec = Task[]
    for (idx,pid) in enumerate(all_w)
        b, e = first(chunks[idx]), last(chunks[idx])
        t = Task(()->remotecall_fetch(f, pid, R[b:e]))
        schedule(t)
        push!(w_exec, t)
    end
    a = vcat(Tuple(fetch(t) for t in w_exec)...)
    return a
end