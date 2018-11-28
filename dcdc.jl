using LinearAlgebra, DataFrames,Optim, ForwardDiff, BenchmarkTools,Distributions,
    Expectations, QuantEcon, Statistics, GLM
using Distributions: invsqrt2π, log2π, sqrt2, invsqrt2
using DCDC
using Test
using Distributed, Suppressor
using Plots

begin "Dynamic Decision Process"
    σ₀ = 1;
    β = 0.8;
    nM = 50;
    nT = 5;
    ddc = DynamicDecisionProcess(σ₀,0.8);
    plot(ddc.ValueFn.xdata,ddc.ValueFn.y);
    plot!(ddc.PolicyFn.xdata,ddc.PolicyFn.y);
    data = simulate_ddc(nM,nT,ddc);
end


# The moment condition is
# u'(c_t) = β d_Trans(c_t,x_t) u'(c_t+1)
c_obs = data.action[1];
x_obs = data.state[1];
data.action[:,1]
c_t = [];
c_t1 = [];
x_t = [];
for t = 1:(nT-1)
    global c_t  = vcat(c_t,data.action[:,t]);
    global x_t  = vcat(x_t,data.state[:,t]);
    global c_t1 = vcat(c_t1,data.action[:,t+1])
end



using Flux #Get derivatives
using Flux.Tracker
using Flux.Tracker: update!

ddc.dtrans.(x_t,c_t)
θ₀ = ones(2)  #θ = (β,σ)
Transition = ddc.trans;
dtrans = (x,c) -> Tracker.gradient(Transition,x,c);
dtransx = (x,c) -> Tracker.gradient(Transition,x,c)[1];
R = dtransx.(c_t,x_t);

lb = [0,0];
ub = [1,100];
function _moment(θ)
    (β,σ) = θ;
    util = Utility(σ);
    dutil = (x) -> Tracker.gradient(util,x)[1].data;
    dudc  = dutil.(c_t);
    dudc1 = dutil.(c_t1);

end
