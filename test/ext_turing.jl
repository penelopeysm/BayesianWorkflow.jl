module BWTuringExtTests

using BayesianWorkflow
using FlexiChains
using DynamicPPL
using Random
using Turing
using Test

Turing.setprogress!(false)

@testset "TuringExt" begin
    @model function f()
        x ~ Normal()
        y ~ Normal(x)
    end
    model = f()

    @testset "construction" begin
        bm = BWModel(f(), (; y = 1.0))
        @test bm isa BWModel{<:DynamicPPL.Model}
        @test only(BayesianWorkflow.get_parameter_variables(bm)) == @varname(x)
        @test only(BayesianWorkflow.get_data_variables(bm)) == @varname(y)
        @test bm.data == VarNamedTuple(y = 1.0)
    end

    @testset "predictive" begin
        bm = BWModel(f(), (; y = 1.0))

        # prior predictive check
        for (variables_kwarg, expected_vns) in (
                (:all, [@varname(x), @varname(y)]),
                (:parameters, [@varname(x)]),
                (:data, [@varname(y)]),
            )
            ppc = prior_predictive_check(bm, 20; variables = variables_kwarg)
            @test ppc isa VNChain
            @test Set(FlexiChains.parameters(ppc)) == Set(expected_vns)
        end

        # posterior
        pst = posterior(bm, 1000)
        @test pst isa VNChain
        @test Set(FlexiChains.parameters(pst)) == Set([@varname(x)])

        # posterior predictive checks
        for (variables_kwarg, expected_vns) in (
                (:all, [@varname(x), @varname(y)]),
                (:parameters, [@varname(x)]),
                (:data, [@varname(y)]),
            )
            ppc = posterior_predictive_check(bm, pst; variables = variables_kwarg)
            @test ppc isa VNChain
            @test Set(FlexiChains.parameters(ppc)) == Set(expected_vns)
        end
    end
end

end # module
