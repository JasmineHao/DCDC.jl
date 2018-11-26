using Pkg
Pkg.activate(".")

# include("./src/DCDC.jl")
# using DCDC
include("./src/functions.jl")

# The current payoff
# The ApproxFn
#_______________________________________________________________
# Linear
# The DynamicDecisionProcess
σ = 1; #Elasticity
β = 0.95;
ddc = DynamicDecisionProcess(σ,β)
using GR
using Plots
gr()
# plotly()
# plotlyjs()
# pyplot()
#gr()
#plotlyjs()
#pgfplots() not tested due to installation issue
GR.plot(ddc.ValueFn.x,ddc.ValueFn.y)
GR.plot(ddc.ValueFn.x,ddc.policy.(ddc.ValueFn.x))

nM = 30
nT = 10
stateCDF = Uniform(minimum(ddc.ValueFn.x),maximum(ddc.ValueFn.x))
x_obs = rand(stateCDF,nM)
a_obs = ddc.policy.(x_obs)
# Define utility
