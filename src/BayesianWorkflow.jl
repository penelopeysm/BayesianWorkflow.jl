module BayesianWorkflow

include("bwmodel.jl")
include("predict.jl")

export BWModel,
    prior_predictive_check,
    posterior,
    posterior_predictive_check

end # module BayesianWorkflow
