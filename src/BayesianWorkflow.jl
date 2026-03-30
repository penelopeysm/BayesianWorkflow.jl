module BayesianWorkflow

include("bwmodel.jl")
include("predict.jl")

using Reexport: @reexport
@reexport using Turing

export BWModel,
    prior_predictive_check,
    posterior,
    posterior_predictive_check

end # module BayesianWorkflow
