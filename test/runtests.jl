using Test

@testset verbose = true "BayesianWorkflow.jl" begin
    include("ext_turing.jl")
end
