import DynamicPPL: DynamicPPL, Model, VarNamedTuple, VarName
import AbstractMCMC: AbstractSampler
import FlexiChains: VNChain
import Random: AbstractRNG

"""
    prior_predictive_check(
        [rng::Random.AbstractRNG,]
        model::BWModel,
        N::Int;
        nchains::Int=1,
        kwargs...
    )::FlexiChains.VNChain
"""
function prior_predictive_check(
        rng::AbstractRNG,
        bm::BWModel,
        N::Int;
        nchains::Int = 1,
        kwargs...
    )
    nchains <= 0 && throw(ArgumentError("nchains must be a positive integer"))
    return if nchains == 1
        sample(rng, bm.model, Prior(), N; kwargs..., chain_type = VNChain)
    else
        sample(rng, bm.model, Prior(), MCMCThreads(), N, nchains; kwargs..., chain_type = VNChain)
    end
end
prior_predictive_check(bm::BWModel, N::Int; nchains::Int = 1, kwargs...) = prior_predictive_check(Random.default_rng(), bm, N; nchains = nchains, kwargs...)

"""
    posterior(
        [rng::Random.AbstractRNG,]
        model::BWModel,
        N::Int;
        sampler::AbstractSampler=Turing.NUTS(),
        nchains::Int=1,
        kwargs...
    )::FlexiChains.VNChain
"""
function posterior(
        rng::AbstractRNG,
        bm::BWModel,
        N::Int;
        sampler::AbstractSampler = NUTS(),
        nchains::Int = 1,
        kwargs...
    )
    cond_model = DynamicPPL.condition(bm.model, bm.data)
    nchains <= 0 && throw(ArgumentError("nchains must be a positive integer"))
    # verbose=false to shut NUTS up
    return if nchains == 1
        sample(rng, cond_model, sampler, N; verbose = false, kwargs..., chain_type = VNChain)
    else
        sample(rng, cond_model, sampler, MCMCThreads(), N, nchains; verbose = false, kwargs..., chain_type = VNChain)
    end
end
posterior(bm::BWModel, N::Int; sampler::AbstractSampler = NUTS(), nchains::Int = 1, kwargs...) = posterior(Random.default_rng(), bm, N; sampler = sampler, nchains = nchains, kwargs...)

"""
    posterior_predictive_check(
        [rng::Random.AbstractRNG,]
        model::BWModel,
        chain::VNChain
    )::FlexiChains.VNChain
"""
function posterior_predictive_check(
        rng::AbstractRNG,
        bm::BWModel,
        chain::VNChain
    )
    return DynamicPPL.predict(rng, bm.model, chain)
end
posterior_predictive_check(bm::BWModel, chain::VNChain) = posterior_predictive_check(Random.default_rng(), bm, chain)
