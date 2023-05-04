module Misc
using JLD2
using Random
using Distributed
using DataStructures: DefaultDict

function get_bit_string(n)
    b = Matrix{Int}(undef, n, 2^n)
    for i = 0:2^(n)-1
        s = bitstring(i)
        b[:, i+1] .= [Int(i)-48 for i in s[length(s)-n+1:end]]
    end
    return b
end

function dict_to_string(a::Dict)
    result = ""
    
    for (i, (key, value)) in enumerate(a)
        
        if i != 1
            result *= "_"
        end
        
        result *= "$(key)=$(value)"
    end
    
    return result
end

function Base.get(x::Vector, i::Int, default)
    if isassigned(x, i)
        return x[i]
    else
       return default 
    end
end

# Get ancillas indices
function get_ancilla_indices(N_state::Int, ancilla_frequency::Int)
    N = N_state * (ancilla_frequency + 1) - ancilla_frequency
    ancilla_indices = [i for i in 1:N if mod1(i, ancilla_frequency+1)!=1]
    state_indices = [i for i in 1:N if mod1(i, ancilla_frequency+1)==1]
    return state_indices, ancilla_indices, N
end

function get_ancilla_indices(N_state::Int, pattern::Vector{Bool})
    @assert any(pattern) "There is no physical states"
    ancilla_indices = Vector{Int}()
    state_indices = Vector{Int}()

    l = length(pattern)
    N = 1
    while true
        if pattern[mod1(N, l)]
            push!(state_indices, N)
        else
            push!(ancilla_indices, N)
        end
        

        if length(state_indices) >= N_state && mod1(N, l) == l
            return state_indices, ancilla_indices, N
        end

        N += 1
    end
    
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
function load_dir(dir; parameter=nothing, parameter_list=false)
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
        if parameter_list
            dd = DefaultDict(Vector)
            for (key, value) in d
                push!(dd[value["params"][parameter]], value)
            end
            d = dd
        else
            
            d = Dict(value["params"][parameter] => value for (file, value) in d)
        end

        return sort(collect(keys(d))), d
    else
        d = Dict(file[1:end-5] => load("$dir/$file") for file in files if file[end-4:end]==".jld2")
    end
    
    return d
end


end