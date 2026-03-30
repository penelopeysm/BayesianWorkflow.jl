import DynamicPPL: DynamicPPL, Model, VarNamedTuple, VarName
import AbstractMCMC: AbstractSampler
import FlexiChains: VNChain
import Random

"""
    prior_predictive_check([rng::Random.AbstractRNG,] bm::BWModel, N::Int; nchains=1, kwargs...)

Sample `N` draws from the full generative model (i.e., from the joint prior over both
parameters and data). Returns a `VNChain` containing draws of all variables in the model.

This is useful for checking whether the prior implies plausible data: inspect the draws of
the data variables to see if they fall within a reasonable range.

# Positional arguments
- `rng`: Random number generator. Defaults to `Random.default_rng()`.
- `bm`: A [`BWModel`](@ref) wrapping the model and data.
- `N`: Number of draws.

# Keyword arguments
- `nchains`: Number of independent chains to sample. Defaults to 1.
- `variables`: Which variables to include in the returned chain. Can be `:all`,
  `:parameters`, or `:data` (default).
- `kwargs...`: Additional keyword arguments passed to `AbstractMCMC.sample`.
"""
function prior_predictive_check(
        rng::Random.AbstractRNG,
        bm::BWModel,
        N::Int;
        nchains::Int = 1,
        variables::Symbol = :data,
        kwargs...
    )
    nchains <= 0 && throw(ArgumentError("nchains must be a positive integer"))
    chn = if nchains == 1
        sample(rng, bm.model, Prior(), N; kwargs..., chain_type = VNChain)
    else
        sample(rng, bm.model, Prior(), MCMCThreads(), N, nchains; kwargs..., chain_type = VNChain)
    end
    return _filter_chain(chn, bm, variables)
end
prior_predictive_check(bm::BWModel, N::Int; nchains::Int = 1, kwargs...) = prior_predictive_check(Random.default_rng(), bm, N; nchains = nchains, kwargs...)

"""
    posterior([rng::Random.AbstractRNG,] bm::BWModel, N::Int; sampler=NUTS(), nchains=1, kwargs...)

Draw `N` samples from the posterior distribution of the model parameters, conditioned on the
data in `bm`. Returns a `VNChain` containing posterior draws of the parameters only.

# Positional arguments
- `rng`: Random number generator. Defaults to `Random.default_rng()`.
- `bm`: A [`BWModel`](@ref) wrapping the model and data.
- `N`: Number of posterior draws per chain.

# Keyword arguments
- `sampler`: The sampler to use. Defaults to `NUTS()`.
- `nchains`: Number of independent chains to sample. Defaults to 1.
- `kwargs...`: Additional keyword arguments passed to `AbstractMCMC.sample`.
"""
function posterior(
        rng::Random.AbstractRNG,
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
    posterior_predictive_check([rng::Random.AbstractRNG,] bm::BWModel, chain::VNChain)

Generate predicted data from the model using posterior parameter draws from `chain`. For
each posterior draw, the data variables are re-sampled from the model, giving a `VNChain` of
predicted data that can be compared against the observed data in `bm.data`.

This is useful for checking whether the fitted model can reproduce key features of the
observed data.

# Positional arguments
- `rng`: Random number generator. Defaults to `Random.default_rng()`.
- `bm`: A [`BWModel`](@ref) wrapping the model and data.
- `chain`: Posterior samples, typically obtained from [`posterior`](@ref).

# Keyword arguments
- `variables`: Which variables to include in the returned chain. Can be `:all`,
  `:parameters`, or `:data` (default).
"""
function posterior_predictive_check(
        rng::Random.AbstractRNG,
        bm::BWModel,
        posterior_chain::VNChain;
        variables::Symbol = :data
    )
    chn = DynamicPPL.predict(rng, bm.model, posterior_chain)
    return _filter_chain(chn, bm, variables)
end
posterior_predictive_check(bm::BWModel, posterior_chain::VNChain) = posterior_predictive_check(Random.default_rng(), bm, posterior_chain)

"""
    _filter_chain(chain::VNChain, bm::BWModel, variables::Symbol)

Helper function to filter a `VNChain` to only include parameters, data, or all variables.
The information about which variables are data vs parameters is taken from the `BWModel`.
"""
function _filter_chain(chain::VNChain, bm::BWModel, variables::Symbol)
    variables == :all && return chain
    data_keys = keys(bm.data)
    keep = if variables == :parameters
        setdiff(bm.all_variables, data_keys)
    elseif variables == :data
        data_keys
    else
        throw(ArgumentError("`variables` must be :all, :parameters, or :data"))
    end
    return chain[collect(keep)]
end
