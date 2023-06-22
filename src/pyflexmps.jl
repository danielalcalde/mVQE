module pyflexmps

using ITensors
using PyCall
using SymPy

const pfs = PyNULL()
const np = PyNULL()
defined_operators = []

function __init__()
    copy!(pfs, pyimport("pyflexmps"))
    copy!(np, pyimport("numpy"))
    copy!(defined_operators, [])
    #for (s, m) in pfs.sigma._matrices
    #    define_op(pfs.sigma(s, 1))
    #end
end

function define_op(op; site_type="Qubit", prepend="pfs")
    global defined_operators
    matrix = Array(np.asarray(op.matrix, dtype="float64"))
    name = "$(prepend)_$(op.s)"
    op_type = typeof(OpName(name))
    site_type = typeof(SiteType(site_type))

    if !(name in defined_operators)
        @eval ITensors.op(::$op_type, ::$site_type) = $matrix
        push!(defined_operators, name)
    end
    
end


function add_operator_to_tuple!(op_tuple, op, site; site_type="Qubit", sublattice=1:site+100)
    define_op(op, site_type=site_type)

    op_name = "pfs_$(op.s)"
    push!(op_tuple, op_name)
    push!(op_tuple, sublattice[site + op.i])
    return op_tuple
end

function substitute_couplings(factor, couplings, site_i)
    for (coupling, strengths) in couplings
        strength = strengths[site_i]
        factor = subs(factor, coupling=>strength)
    end
    return factor
end

function add_sympy_to_opsum!(opsum, op, sites::Vector{<:Integer}; couplings=Dict(), sublattice=1:sites[end]+100, site_type="Qubit")
    if op.__class__ == sympy.core.add.Add
        for opi in op.args
            opsum = add_sympy_to_opsum!(opsum, opi, sites; couplings=couplings, sublattice=sublattice, site_type=site_type)
        end
    else    
        op, factor_ = pfs.quantum_operators.seperate_factor(op)
        for (site_i, site) in enumerate(sites)
            op_tuple = []
            
            if op.__class__ == sympy.core.mul.Mul
                for opi in op.args
                    op_tuple = add_operator_to_tuple!(op_tuple, opi, site; sublattice=sublattice, site_type=site_type)
                end
            else
                op_tuple = add_operator_to_tuple!(op_tuple, op, site; site_type=site_type, sublattice=sublattice)
            end

            factor_c = substitute_couplings(factor_, couplings, site_i)
            factor_c = SymPy.N(factor_c)

            @assert !(factor_c isa Sym)
            @assert factor_c isa Number

            if factor_c != 0
                pushfirst!(op_tuple, factor_c)
                op_tuple = tuple(op_tuple...)
                opsum += op_tuple
            end
        end
    end
    return opsum
end


function convert_sympy_to_opsum(sympy_ham, sites::Vector{<:Integer}; kwargs...)
    sympy_ham = pfs.apply(expand(sympy_ham))
    return add_sympy_to_opsum!(OpSum(), sympy_ham, sites; kwargs...)
end

function convert_sympy_to_opsums(sympy_ham, sites::Vector{<:Integer}; kwargs...)
    sympy_ham = pfs.apply(expand(sympy_ham))
    return [add_sympy_to_opsum!(OpSum(), sympy_ham, [site]; kwargs...) for site in sites]
end



end # module