module Misc
using JLD2
using Random
using Distributed
using DataStructures: DefaultDict

struct ExponentialIterator{T<:Number}
    n1::T
    n2::T
    k::Integer
    skip_first::Bool
    ExponentialIterator(n1::T, n2::T, k; skip_first::Bool=false) where T<:Number = new{T}(n1, n2, k, skip_first)
        
end
Base.length(x::ExponentialIterator) = x.k
function Base.iterate(x::ExponentialIterator, state::Int=0)
    if state >= x.k
        return nothing
    end
    return (get_elem(x, state), state + 1)
end

function get_elem(x::ExponentialIterator{T}, i::Integer) where T
    i += x.skip_first
    k = x.k + x.skip_first
    n = x.n1 * (x.n2 / x.n1) ^ (i / (k-1))
    if T <: Integer
        n = Integer(round(n))
    end
    return n
end

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
"""    
load_dir(dir; parameter=nothing, parameter_list=false, sort_parameter=true, parameter_type_func=x->x)

Load a directory of files. If parameter is specified, the files are sorted according to the parameter. If parameter_list is true, the files are grouped according to the parameter. If sort_parameter is true, the parameter is sorted. If parameter_type_func is specified, the parameter is converted using the function before sorting.
"""
function load_dir(dir; parameter=nothing, parameter_list=false, sort_parameter=true, parameter_type_func=x->x)
    files = readdir(dir)
    
    if parameter isa Vector
        if  !(parameter_type_func isa Vector)
            parameter_type_func = [parameter_type_func for _ in 1:length(parameter)]
        end
        
        d = Dict()
        if parameter_list
            dd = DefaultDict(Vector)
            for file in files
                if file[end-4:end]==".jld2"
                    data = load("$dir/$file")
                    params = [parameter_type_func_(data["params"][para]) for (parameter_type_func_, para) in zip(parameter_type_func, parameter)]
                    push!(dd[params], data)
                end
            end
            d = dd
        
        else
            for file in files
                if file[end-4:end]==".jld2"
                    data = load("$dir/$file")
                    params = [parameter_type_func_(data["params"][para]) for (parameter_type_func_, para) in zip(parameter_type_func, parameter)]
                    d[params] = data
                end
            end
        end
        
        parameter_set = [Set() for _ in 1:length(parameter)]
        for key in keys(d)
            for i in 1:length(parameter)
                parameter_set[i] = push!(parameter_set[i], key[i])
            end
        end
        
        if sort_parameter
            paramter_vec = [sort(collect(s)) for s in parameter_set]
        else
            paramter_vec = [collect(s) for s in parameter_set]
        end

        return paramter_vec, d
        
        
    elseif parameter !== nothing
        d = Dict(file[1:end-5] => load("$dir/$file") for file in files if file[end-4:end]==".jld2")
        if parameter_list
            dd = DefaultDict(Vector)
            for (key, value) in d
                parameter_value = parameter_type_func(value["params"][parameter])
                push!(dd[parameter_value], value)
            end
            d = dd
        else
            
            d = Dict(parameter_type_func(value["params"][parameter]) => value for (file, value) in d)
        end
        params = collect(keys(d))

        if sort_parameter
            params = sort(params)
        end
        return params, d
    else
        d = Dict(file[1:end-5] => load("$dir/$file") for file in files if file[end-4:end]==".jld2")
    end
    
    return d
end

function simple_spinhalf_to_spin1_vec(spinhalf_vec)
    spin1_vec = Vector{Float64}(undef, length(spinhalf_vec)÷2)
    for i in 1:length(spin1_vec)
        if spinhalf_vec[2i-1] == 1 && spinhalf_vec[2i] == 1
            spin1_vec[i] = 0
        elseif spinhalf_vec[2i-1] == 2 && spinhalf_vec[2i] == 1
            spin1_vec[i] = 1
        elseif spinhalf_vec[2i-1] == 1 && spinhalf_vec[2i] == 2
            spin1_vec[i] = -1
        else
            spin1_vec[i] = 100
        end
    end
    return spin1_vec
end


end