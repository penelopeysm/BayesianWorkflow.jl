using Test
using BayesianWorkflow
using Turing

@testset verbose = true "BayesianWorkflow.jl" begin
    include("ext_turing.jl")
end
