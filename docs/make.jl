using Documenter
using DocumenterInterLinks
using BayesianWorkflow

links = InterLinks(
    # "DynamicPPL" => "https://turinglang.org/DynamicPPL.jl/stable/",
    "Julia" => "https://docs.julialang.org/en/v1/",
)

modules = [
    BayesianWorkflow,
]

makedocs(;
    sitename = "BayesianWorkflow.jl",
    modules = modules,
    pages = [
        "index.md",
        "api.md",
    ],
    checkdocs = :exports,
    warnonly = true,
    doctest = false,
    plugins = [links],
)
