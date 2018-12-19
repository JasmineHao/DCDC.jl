begin
    using LinearAlgebra, DataFrames,Optim, ForwardDiff, BenchmarkTools,Distributions,
        Expectations, QuantEcon, Statistics, GLM
    using Distributions: invsqrt2π, log2π, sqrt2, invsqrt2
    using DCDC
    using Test
    using Distributed, Suppressor
    using Plots
end

begin "Dynamic Decision Process"
    σ₀ = 1;
    β = 0.95;
    α = 0.7;
    nM = 50;
    nT = 2;
    ddc = DynamicDecisionProcess(σ₀,β,α);
    computeEquilibrium(ddc);
    ee = EulerEquation(σ₀,β,α);
end
begin
    # data = simulate_ddc(nM,nT,ddc);
    data =simulate_ee(nM,nT,ee);
end

plot(ddc.ValueFn.xdata[:,1],ddc.ValueFn.y);
plot(ddc.PolicyFn.xdata[:,1],ddc.PolicyFn.y);
plot!(ee.PolicyFn.xdata[:,1],ee.PolicyFn.y);
scatter(ddc.PolicyFn.xdata[:,2]);
checked = [check_ee(ddc) for i = 1:100];
scatter(checked[checked.<2])

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


lb = [0,0];
ub = [1,100];

function _moment(θ)
    (β,σ) = θ;
    util = Utility(σ);
    dutil = (x) -> Tracker.gradient(util,x)[1].data;
    dudc  = dutil.(c_t);
    dudc1 = dutil.(c_t1);
    dtrans=ee.dtrans;
    dtrans_s = (c,s) -> dtrans(c,s)[2].data;

    return(dudc - β .* dtrans_s.(c_t,x_t) .* dudc1)
end


ϵ=_moment((0.8,1))
ϵ' * ϵ

function f_obj(θ)
    ϵ=_moment(θ);
    return(ϵ' * ϵ);
end


result = optimize(f_obj,[0.0001,0.01],[0.9999,10.0],[0.8,3.5])
result.minimizer


# Comment: Obviously the Euler equation approach cannot be used....
# BUT WHY?!
