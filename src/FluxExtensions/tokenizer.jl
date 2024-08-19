
function simple_tokenizer(M::Vector, tokenization_range; tokenization_interval=tokenization_range)
    M = Int.(M)
    space_size = 2 ^ tokenization_range
    k = length(M) ÷ tokenization_interval
    if mod(length(M), tokenization_interval) != 0
       k += 1
    end
    v = Matrix{Int}(undef, k, space_size)
    for (i, j) in enumerate(1:tokenization_interval:length(M))
        u = j + tokenization_range - 1
        v[i, :] = elem_tokenizer(M[j:min(u,length(M))]; space_size)
    end
    return v
end

function simple_tokenizer(M::Matrix, tokenization_range; tokenization_interval=tokenization_range)
    v0 = simple_tokenizer(M[:, 1], tokenization_range; tokenization_interval=tokenization_interval)
    
    v = Array{Int, 3}(undef, size(v0, 1), size(v0, 2), size(M, 2))
    v[:, :, 1] = v0
    for i in 2:size(M, 2)
        v[:, :, i] = simple_tokenizer(M[:, i], tokenization_range; tokenization_interval=tokenization_interval)
    end
    return v
end


function simple_tokenizer(M::Array{T, 3}, tokenization_range; tokenization_interval=tokenization_range) where T
    @assert size(M, 2) == 1
    return simple_tokenizer(M[:, 1, :], tokenization_range; tokenization_interval=tokenization_interval)
end

function elem_tokenizer(mi; space_size=2^length(mi))
    v = zeros(Int, space_size)
    elem = parse(Int, join(string.(reverse(mi))); base=2) + 1
    v[elem] = 1
    return v
end

struct SpinTokenizer
    tokenization_range::Int
    tokenization_interval::Int
end

SpinTokenizer(tokenization_range::Int; tokenization_interval=tokenization_range) = SpinTokenizer(tokenization_range, tokenization_interval)
(f::SpinTokenizer)(M) = Zygote.@ignore simple_tokenizer(M, f.tokenization_range; tokenization_interval=f.tokenization_interval)

struct PermuteDims
    permutation::Tuple
end

function (f::PermuteDims)(M)
    permutation = f.permutation[1:ndims(M)]
    permutedims(M, permutation)
end
struct ReshapeDims
    shape::Tuple
end

(f::ReshapeDims)(M) = reshape(M, f.shape)