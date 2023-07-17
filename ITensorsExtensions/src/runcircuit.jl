function runcircuit(
   M::MPS,
   circuit::Union{Tuple,AbstractVector};
   full_representation::Bool=false,
   noise=nothing,
   eltype=nothing,
   apply_dag=nothing,
   cutoff=1e-15,
   maxdim=10_000,
   svd_alg="divide_and_conquer",
   move_sites_back::Bool=true,
   device=identity,
   gate_grad=true,
   onequbit_gates=false,
   unitary=false
   )
   @assert noise === nothing "Noise is not implemented yet."

   M = full_representation ? PastaQ.convert_to_full_representation(M) : M
   circuit_tensors = PastaQ.buildcircuit(M, circuit; noise, eltype, device)
   if gate_grad
      if onequbit_gates
         return apply_onequbit(circuit_tensors, M; unitary)
      else
         return apply(circuit_tensors, M; cutoff, maxdim, svd_alg, move_sites_back)
      end
            
   else
      return apply_nogategrad(circuit_tensors, M; cutoff, maxdim, svd_alg, move_sites_back)
   end
end