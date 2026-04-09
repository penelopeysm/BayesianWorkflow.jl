module BayesianWorkflowTuringExt

import BayesianWorkflow: BayesianWorkflow, BWModel
using Turing
import DynamicPPL: DynamicPPL, VarNamedTuple, VarName, Model
import Random
import OrderedCollections: OrderedSet
import AbstractMCMC: AbstractSampler
import FlexiChains: VNChain

const TuringBWModel = BWModel{<:DynamicPPL.Model}

function BayesianWorkflow.to_bwmodel(model::DynamicPPL.Model, data = nothing)
    # Run model once to get parameters.
    vnt = rand(model)
    return model, _to_vnt(data, vnt), OrderedSet(keys(vnt))
end

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

function BayesianWorkflow.sample_prior(
        rng::Random.AbstractRNG,
        bm::TuringBWModel,
        N::Int,
        nchains::Int;
        kwargs...
    )
    return if nchains == 1
        sample(rng, bm.model, Prior(), N; kwargs..., chain_type = VNChain)
    else
        sample(rng, bm.model, Prior(), MCMCThreads(), N, nchains; kwargs..., chain_type = VNChain)
    end
end

function BayesianWorkflow.sample_posterior(
        rng::Random.AbstractRNG,
        bm::TuringBWModel,
        sampler,
        N::Int,
        nchains::Int;
        kwargs...
    )
    cond_model = DynamicPPL.condition(bm.model, bm.data)
    # verbose=false to shut NUTS up
    return if nchains == 1
        sample(rng, cond_model, sampler, N; verbose = false, kwargs..., chain_type = VNChain)
    else
        sample(rng, cond_model, sampler, MCMCThreads(), N, nchains; verbose = false, kwargs..., chain_type = VNChain)
    end
end
BayesianWorkflow.default_sampler(::TuringBWModel) = Turing.NUTS()

function BayesianWorkflow.predict(rng::Random.AbstractRNG, bm::TuringBWModel, posterior_chain::VNChain)
    return DynamicPPL.predict(rng, bm.model, posterior_chain)
end

end # module
