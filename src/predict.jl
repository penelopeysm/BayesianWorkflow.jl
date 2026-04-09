import FlexiChains: FlexiChain
import AbstractMCMC: AbstractSampler
import Random

"""
    prior_predictive_check([rng::Random.AbstractRNG,] bm::BWModel, N::Int; nchains=1, kwargs...)

Sample `N` draws from the full generative model (i.e., from the joint prior over both
parameters and data). Returns a `FlexiChain` containing draws of all variables in the model.

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
    chn = sample_prior(rng, bm, N, nchains; kwargs...)
    return _filter_chain(chn, bm, variables)
end
prior_predictive_check(bm::BWModel, N::Int; nchains::Int = 1, kwargs...) = prior_predictive_check(Random.default_rng(), bm, N; nchains = nchains, kwargs...)

# To be overloaded by extensions
# TODO: Document
function sample_prior end

"""
    posterior([rng::Random.AbstractRNG,] bm::BWModel, N::Int; sampler=default_sampler(bm), nchains=1, kwargs...)

Draw `N` samples from the posterior distribution of the model parameters, conditioned on the
data in `bm`. Returns a `FlexiChain` containing posterior draws of the parameters only.

# Positional arguments
- `rng`: Random number generator. Defaults to `Random.default_rng()`.
- `bm`: A [`BWModel`](@ref) wrapping the model and data.
- `N`: Number of posterior draws per chain.

# Keyword arguments
- `sampler`: The sampler to use. By default this looks up `BayesianWorkflow.default_sampler(bm)`.
- `nchains`: Number of independent chains to sample. Defaults to 1.
- `kwargs...`: Additional keyword arguments passed to `AbstractMCMC.sample`.
"""
function posterior(
        rng::Random.AbstractRNG,
        bm::BWModel,
        N::Int;
        sampler = default_sampler(bm),
        nchains::Int = 1,
        kwargs...
    )
    nchains <= 0 && throw(ArgumentError("nchains must be a positive integer"))
    return sample_posterior(rng, bm, sampler, N, nchains; kwargs...)
end
posterior(bm::BWModel, N::Int; sampler = default_sampler(bm), nchains::Int = 1, kwargs...) = posterior(Random.default_rng(), bm, N; sampler = sampler, nchains = nchains, kwargs...)

# To be overloaded by extensions
# TODO: Document
# TODO: default_sampler as a struct works for Turing, but might need to be generalised for
# other models, because there isn't as clean a notion of 'sampler'.
function sample_posterior end
function default_sampler end

"""
    posterior_predictive_check([rng::Random.AbstractRNG,] bm::BWModel, chain::FlexiChain)

Generate predicted data from the model using posterior parameter draws from `chain`. For
each posterior draw, the data variables are re-sampled from the model, giving a `FlexiChain`
of predicted data that can be compared against the observed data in `bm.data`.

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
        posterior_chain::FlexiChain;
        variables::Symbol = :data
    )
    chn = predict(rng, bm, posterior_chain)
    return _filter_chain(chn, bm, variables)
end
posterior_predictive_check(bm::BWModel, posterior_chain::FlexiChain; kwargs...) = posterior_predictive_check(Random.default_rng(), bm, posterior_chain; kwargs...)

# To be overloaded
# TODO: Document
function predict end

"""
    _filter_chain(chain::FlexiChain, bm::BWModel, variables::Symbol)

Helper function to filter a `FlexiChain` to only include parameters, data, or all variables.
The information about which variables are data vs parameters is taken from the `BWModel`.
"""
function _filter_chain(chain::FlexiChain, bm::BWModel, variables::Symbol)
    variables == :all && return chain
    keep = if variables == :parameters
        get_parameter_variables(bm)
    elseif variables == :data
        get_data_variables(bm)
    else
        throw(ArgumentError("`variables` must be :all, :parameters, or :data"))
    end
    return chain[collect(keep)]
end
