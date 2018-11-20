using Pkg
Pkg.activate(".")
# include("./src/DCDC.jl")
using DCDC

# The current payoff
# The ApproxFn
#_______________________________________________________________
# Linear
# The DynamicDecisionProcess
σ = 1; #Elasticity
β = 0.95;
ddc = DynamicDecisionProcess(1,0.95)
# Define utility
using Plots
plot(ddc.ValueFn.x,ddc.ValueFn.y)
plot!(ddc.ValueFn.x,ddc.policy)
