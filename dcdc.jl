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
    α = 1.2;
    ddc = DynamicDecisionProcess(σ₀,β,α);
    computeEquilibrium(ddc);
    # ee = EulerEquation(σ₀,β,α);
end
begin
    nM = 50;
    nT = 200;
    data = simulate_ddc(nM,nT,ddc);
end
begin
    # data =simulate_ee(nM,nT,ee);
    # The moment condition is
    # u'(c_t) = β d_Trans(c_t,x_t) u'(c_t+1)
    c_t = [];
    c_t1 = [];
    x_t = [];
    for t = 1:(nT-1)
        global c_t  = vcat(c_t,data.action[:,t]);
        global x_t  = vcat(x_t,data.state[:,t]);
        global c_t1 = vcat(c_t1,data.action[:,t+1])
    end
end

scatter(ddc.ValueFn.xdata[:,1],ddc.ValueFn.y);
scatter(ddc.PolicyFn.xdata[:,1],ddc.PolicyFn.y);
# plot!(ee.PolicyFn.xdata[:,1],ee.PolicyFn.y);
# scatter(ddc.PolicyFn.xdata[:,2]);
# checked = [check_ee(ddc) for i = 1:100];
# scatter(checked[checked .< 100])
#

using Flux #Get derivatives
using Flux.Tracker
using Flux.Tracker: update!


lb = [0,0];
ub = [1,100];

function f_obj(θ)
    ϵ=_moment(θ);
    return(ϵ' * ϵ);
end

result = optimize(f_obj,[0.0001,0.01],[0.9999,10.0],[0.8,3.5])
result.minimizer


# Comment: Obviously the Euler equation approach cannot be used....
# BUT WHY?!
# Results of Optimization Algorithm
#  * Algorithm: Fminbox with L-BFGS
#  * Starting Point: [0.8,3.5]
#  * Minimizer: [0.7181627130538114,0.010000000000000021]
#  * Minimum: 6.840716e+00
#  * Iterations: 11
#  * Convergence: true
#    * |x - x'| ≤ 0.0e+00: true
#      |x - x'| = 0.00e+00
#    * |f(x) - f(x')| ≤ 0.0e+00 |f(x)|: true
#      |f(x) - f(x')| = 0.00e+00 |f(x)|
#    * |g(x)| ≤ 1.0e-08: false
#      |g(x)| = 8.84e+00
#    * Stopped by an increasing objective: false
#    * Reached Maximum Number of Iterations: false
#  * Objective Calls: 2495
#  * Gradient Calls: 2495

#_____________________________________________________________________________
# Secition 2 : Estimate η
#_____________________________________________________________________________
