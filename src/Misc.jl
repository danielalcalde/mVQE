module Misc

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

# Print dictionary in a nice way
function pprint(d::Dict)
    for (k, v) in d
        println("$k => $v")
    end
end

end