module Misc
using JLD2

function get_bit_string(n)
    b = Matrix{Int}(undef, n, 2^n)
    for i = 0:2^(n)-1
        s = bitstring(i)
        b[:, i+1] .= [Int(i)-48 for i in s[length(s)-n+1:end]]
    end
    return b
end

function Base.get(x::Vector, i::Int, default)
    if isassigned(x, i)
        return x[i]
    else
       return default 
    end
end

# Get ancillas indices
function get_ancillas_indices(N_state::Int, ancilla_frequency::Int)
    N = N_state * (ancilla_frequency + 1) - ancilla_frequency
    ancillas_indices = [i for i in 1:N if mod1(i, ancilla_frequency+1)!=1]
    state_indices = [i for i in 1:N if mod1(i, ancilla_frequency+1)==1]
    return state_indices, ancillas_indices, N
end

"""
    unravel_index(index::Int, shape::Tuple})

Unravel an index into a tuple of indices.
Example:
    unravel_index(1, (2, 3)) = (1, 1)
    unravel_index(2, (2, 3)) = (2, 1)
    unravel_index(3, (2, 3)) = (1, 2)
    unravel_index(5, (2, 3)) = (1, 3)
"""
function unravel_index(index::Int, shape::Tuple)
    @assert index <= prod(shape)
    index -= 1
    indices = Vector{Int16}(undef, length(shape))
    for (i, si) in enumerate(shape)
        indices[i] = mod(index, si)
        index = Int((index - indices[i]) / si)
    end
    @assert index == 0 "$index"
    return indices .+ 1
end

# Print dictionary in a nice way
function pprint(d::Dict)
    for (k, v) in d
        println("$k => $v")
    end
end

# Load a directory of files
function load_dir(dir; parameter=nothing)
    files = readdir(dir)
    
    
    if parameter isa Vector
        d = Dict()
                
        for file in files
            if file[end-4:end]==".jld2"
                data = load("$dir/$file")
                params = [data["params"][para] for para in parameter]
                d[params] = data
            end
        end 
        
        parameter_set = [Set() for _ in 1:length(parameter)]
        for key in keys(d)
            for i in 1:length(parameter)
                parameter_set[i] = push!(parameter_set[i], key[i])
            end
        end
        paramter_vec = [sort(collect(s)) for s in parameter_set]
        
        return paramter_vec, d
        
        
    elseif parameter !== nothing
        d = Dict(file[1:end-5] => load("$dir/$file") for file in files if file[end-4:end]==".jld2")
        d = Dict(value["params"][parameter] => value for (file, value) in d)
        return sort(collect(keys(d))), d
    else
        d = Dict(file[1:end-5] => load("$dir/$file") for file in files if file[end-4:end]==".jld2")
    end
    
    return d
end


end