module DistributedExtensions
using Distributed

function remotecall_eval_async(m::Module, procs, ex)
    
    run_locally = 0
    for pid in procs
        if pid == myid()
            run_locally += 1
        else
            Distributed.@async_unwrap remotecall_wait(Core.eval, pid, m, ex)
        end
    end
    yield() # ensure that the remotecalls have had a chance to start

    # execute locally last as we do not want local execution to block serialization
    # of the request to remote nodes.
    for _ in 1:run_locally
        @async Core.eval(m, ex)
    end

nothing
end

macro everywhere_async(ex)
    procs = GlobalRef(@__MODULE__, :procs)
    return esc(:($(mVQE.DistributedExtensions).@everywhere_async $procs() $ex))
end

macro everywhere_async(procs, ex)
    imps = Distributed.extract_imports(ex)
    return quote
        $(isempty(imps) ? nothing : Expr(:toplevel, imps...)) # run imports locally first
        let ex = Expr(:toplevel, :(task_local_storage()[:SOURCE_PATH] = $(get(task_local_storage(), :SOURCE_PATH, nothing))), $(esc(Expr(:quote, ex)))),
            procs = $(esc(procs))
            remotecall_eval_async(Main, procs, ex)
        end
    end
end

end # module