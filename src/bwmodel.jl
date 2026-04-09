import OrderedCollections: OrderedSet

"""
    BWModel(model, data)

A wrapper that stores a model together with its associated data.

`model` is intended to be backend-agnostic. However, right now, only a Turing.jl model
(i.e., `model::DynamicPPL.Model`) is accepted.

## Turing.jl

The model itself should be written in a way that does *not* hardcode data variables
in the model definition. For example, if `y` here is meant to be data:

```julia
@model function mymodel()
    x ~ Normal()
    y ~ Normal(x)
end
```

then `y` should not be specified as an argument, nor should `mymodel()` be conditioned
on `y`. Instead, just pass `mymodel()` to `BWModel` and then provide the data for `y`
as the second argument.
"""
struct BWModel{M, D, K}
    model::M
    data::D
    # Note that `all_variables` includes data variables too
    all_variables::OrderedSet{K}

    # This single inner constructor is a way to have backend-specific inner constructors:
    # extensions have to overload `to_bwmodel` instead of the actual constructor
    function BWModel(model, data = nothing)
        model, data, all_variables = to_bwmodel(model, data)
        M, D, K = typeof(model), typeof(data), eltype(all_variables)
        return new{M, D, K}(model, data, all_variables)
    end
end

"""
    to_bwmodel

The equivalent of an inner constructor for `BWModel`. This can be overloaded by each backend
to provide backend-specific construction of `BWModel`s.

The expected signature is `to_bwmodel(model[, data])`, which should then return three things:

- `model` (probably unchanged)
- `data` (probably unchanged, but can be normalised into a form that is convenient to work with)
- `all_variables` (an `OrderedSet{K}` of variables in the model, including data variables:
  the key type can be chosen by the backend)
"""
function to_bwmodel end

"""
    get_data_variables(::BWModel{M,D,K})

Obtain a collection of data variables. The return value must be a collection with element type `K`.

By default this returns `keys(data)`, but can be overridden by specific backends if necessary.
"""
function get_data_variables(b::BWModel)
    return keys(b.data)
end

"""
    get_parameter_variables(::BWModel{M,D,K})

Obtain a collection of parameter variables. The return value must be a collection with
element type `K`.

By default this returns `setdiff(b.all_variables, get_data_variables(b))`, but this can be
overridden by specific backends if necessary.
"""
function get_parameter_variables(b::BWModel)
    return setdiff(b.all_variables, get_data_variables(b))
end

Base.:(==)(m1::BWModel, m2::BWModel) = (m1.model == m2.model) & (m1.data == m2.data) & (m1.all_variables == m2.all_variables)
Base.isequal(m1::BWModel, m2::BWModel) = isequal(m1.model, m2.model) && isequal(m1.data, m2.data) && isequal(m1.all_variables, m2.all_variables)

function Base.show(io::IO, ::MIME"text/plain", bm::BWModel{M, D, K}) where {M, D, K}
    data_keys = get_data_variables(bm)
    param_keys = get_parameter_variables(bm)
    printstyled(io, "BWModel{$(nameof(M)),$(nameof(D)),$(nameof(K))}"; bold = true)
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
