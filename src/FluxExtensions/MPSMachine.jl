# MPSMachine
struct MPSMachine
    A
    C
end
Flux.@functor MPSMachine
Flux.trainable(self::MPSMachine) = (self.A, self.C)
function MPSMachine(dv::Integer, dh::Integer; d_out::Integer=dh)
    d = Haar(1)
    A = Array{Float64, 3}(undef, dh, dh, dv)
    for i in 1:dv
        A[:, :, i] = rand(d, dh)
    end
    C = rand(d, max(dh, d_out))[1:d_out, 1:dh]
    return MPSMachine(A, C)
end
function (self::MPSMachine)(input::Array{T, 2}) where T
    input = reshape(input, size(input, 1), size(input, 2), 1)
    return self(input)[:, :, 1]
end

function (self::MPSMachine)(input::Array{T, 3}) where T
    h = @Zygote.ignore begin
        h = zeros(size(self.A, 1), size(input, 3))
        h[1, :] .= 1.
        return h
    end
    hs = nothing
   
    @tensor As[i, j, l, b] := self.A[i, j, v] * input[v, l, b]
    
    for i in 1:size(input, 2)
        A = As[:, :, i, :]
        h = NNlib.batched_mul(A, reshape(h, size(h, 1), 1, size(h, 2)))[:, 1, :]
        if hs === nothing
            hs = h
        else
            hs = hcat(hs, h)
        end
    end
    hs = self.C * hs
    hs = reshape(hs, size(self.C, 1), size(input, 3), size(input, 2))
    return permutedims(hs, (1, 3, 2))
end