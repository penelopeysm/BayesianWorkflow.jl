import OrderedCollections: OrderedSet
import DynamicPPL: DynamicPPL, VarNamedTuple, VarName

"""
    _to_vnt(vals)

Convert `vals` to a VarNamedTuple.
"""
_to_vnt(::Nothing, template) = VarNamedTuple()
_to_vnt(n::NamedTuple, template) = VarNamedTuple(n)
function _to_vnt(d::AbstractDict, template)
    # TODO(penelopeysm): This function gets used a fair bit in DynamicPPL. Should
    # probably consider upstreaming it.
    vnt = VarNamedTuple()
    for (vn, val) in pairs(d)
        vn isa VarName || throw(ArgumentError("keys of data dictionary must be VarNames, got $(typeof(vn))"))
        top_sym = AbstractPPL.getsym(vn)
        template = get(template.data, top_sym, DynamicPPL.NoTemplate())
        vnt = DynamicPPL.templated_setindex!!(vnt, val, vn, template)
    end
    return vnt
end
function _to_vnt(v::VarNamedTuple, template)
    # The user might have given a crappy VNT without templates. Since we have the template
    # data, we may as well upgrade it; at worst this is a no-op.
    vnt = VarNamedTuple()
    for (vn, val) in pairs(v)
        top_sym = AbstractPPL.getsym(vn)
        template = get(template.data, top_sym, DynamicPPL.NoTemplate())
        vnt = DynamicPPL.templated_setindex!!(vnt, val, vn, template)
    end
    return vnt
end

"""
    BModel(model::DynamicPPL.Model, data)

A wrapper to store a Turing model and associated data.
"""
struct BWModel{M <: DynamicPPL.Model, V <: VarNamedTuple}
    model::M
    data::V
    # Note that `all_variables` includes data variables too.
    all_variables::OrderedSet{VarName}

    function BWModel(model::M, data = nothing) where {M}
        # Run model once to get parameters.
        vnt = rand(model)
        all_variables = OrderedSet(keys(vnt))
        vnt_data = _to_vnt(data, vnt)
        return new{M, typeof(vnt_data)}(model, vnt_data, all_variables)
    end
end
Base.:(==)(m1::BWModel, m2::BWModel) = (m1.model == m2.model) & (m1.data == m2.data) & (m1.all_variables == m2.all_variables)
Base.isequal(m1::BWModel, m2::BWModel) = isequal(m1.model, m2.model) && isequal(m1.data, m2.data) && isequal(m1.all_variables, m2.all_variables)
function Base.show(io::IO, ::MIME"text/plain", bm::BWModel)
    data_keys = keys(bm.data)
    param_keys = setdiff(bm.all_variables, data_keys)
    printstyled(io, "BWModel"; bold = true)
    println(io)
    printstyled(io, "  Model: "; color = :light_black)
    println(io, bm.model.f)
    # Parameters
    printstyled(io, "  Parameters"; color = :blue, bold = true)
    if isempty(param_keys)
        printstyled(io, " (none)"; color = :light_black)
    else
        for vn in param_keys
            print(io, "\n    ")
            printstyled(io, vn; color = :blue)
        end
    end
    println(io)
    # Data
    printstyled(io, "  Data"; color = :green, bold = true)
    return if isempty(data_keys)
        printstyled(io, " (none)"; color = :light_black)
    else
        for (vn, val) in pairs(bm.data)
            print(io, "\n    ")
            printstyled(io, vn; color = :green)
            printstyled(io, " = "; color = :light_black)
            print(io, val)
        end
    end
end
