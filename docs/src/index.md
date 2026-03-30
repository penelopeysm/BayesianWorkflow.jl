# BayesianWorkflow.jl

Welcome!

```@example intro
using BayesianWorkflow

@model function f()
    x ~ Normal()
    y ~ Normal(x)
end

bm = BWModel(f(), (; y = 1.0))
```

```@example intro
prior_predictive_check(bm, 1000)
```

```@example intro
post = posterior(bm, 1000)
```

```@example intro
posterior_predictive_check(bm, post)
```
