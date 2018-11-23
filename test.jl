using Pkg
Pkg.activate(".")

# include("./src/DCDC.jl")
# using DCDC
include("functions.jl")

# The current payoff
# The ApproxFn
#_______________________________________________________________
# Linear
# The DynamicDecisionProcess
σ = 1; #Elasticity
β = 0.8;
ddc = DynamicDecisionProcess(σ,β)

nM = 30
nT = 10
stateCDF = Uniform(minimum(ddc.ValueFn.x),maximum(ddc.ValueFn.x))
x_obs = rand(stateCDF,nM)

# Define utility

using Plots
gr()
display(Plots.plot(randn(10)))
Plots.plot(ddc.ValueFn.x,ddc.ValueFn.y)
# plot!(ddc.ValueFn.x,ddc.policy)
