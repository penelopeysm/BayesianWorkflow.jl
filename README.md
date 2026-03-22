# BayesianWorkflow.jl

[...]

## Installation

BayesianWorkflow.jl is not on the General registry yet.
You can install it with:

```julia
] add https://github.com/penelopeysm/BayesianWorkflow.jl
```

## Code formatting

BayesianWorkflow.jl uses Runic.jl as its code formatter.

Every PR should be formatted with Runic.jl before merging.
If you want, you can install Runic and manually format the code using [the instructions on the Runic.jl repository](https://github.com/fredrikekre/Runic.jl).

However, it is probably easiest to use `pre-commit` for this.
First, [install `pre-commit`](https://pre-commit.com/) using your package manager of choice.
Then, from the root of the repository, run

```bash
pre-commit install
```

Every time you make a commit, `pre-commit` will now automatically run Runic.jl on the code and format it for you.

To manually format code, you can run

```bash
pre-commit run --all-files
```
