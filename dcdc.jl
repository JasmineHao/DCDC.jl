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
    ρ = 0.2
    β = 1 /(1+ ρ);
    r = 0.05;
    α = 0.7;
    A = 1 + r;
    ddc = DynamicDecisionProcess(σ₀,β,α,A);
    # The log function has a solution
    value_function(s) = (log(1 - β * α) + (β * α) * log(β * α) /(1-β * α) ) / (1 - β) + α * log(s) /(1-β * α);
    policy_function(s) = (1 - β * α) * A * s^α;
    ddc.ValueFn.y = value_function.(ddc.ValueFn.xdata);
    ddc.PolicyFn.y = policy_function.(ddc.PolicyFn.xdata[:,1]);
    computeEquilibrium(ddc);
    # ee = EulerEquation(σ₀,β,α);
end

# Check data simulation
n = 1;
s = Array{State,2}; #nM x nT state
a = Array{State,2}; #nM x nT action
while n < nM
    s[n,1] = State()
    for t = 1:(nT-1)
    # x = hcat(s[:,t],randn(nM)/3);
        x = hcat(s[:,t],rand(du,nM)); #This is a bit problematic
        a[:,t] = ddc.PolicyFn(x);
        s_1 = ddc.trans(a[:,t],s[:,t]);
        if length(s_1 .< 0) > 0
            break;
        end
        s[:,t+1] = s_1;
    end
end

# "Check Approximation of Policy Function"
begin
    scatter(ddc.PolicyFn.xdata[:,1],ddc.PolicyFn.y);
    scatter!(ddc.PolicyFn.xdata[:,1],ddc.PolicyFn(ddc.PolicyFn.xdata));
    scatter!(ddc.PolicyFn.xdata[:,1],policy_function.(ddc.PolicyFn.xdata[:,1]));
    scatter!(ddc.PolicyFn.xdata[:,1],A * ddc.PolicyFn.xdata[:,1].^α)
end

# "Check Approximation of Value Function"
begin
    scatter(ddc.ValueFn.xdata[:,1] ,ddc.ValueFn.y);
    scatter!(ddc.ValueFn.xdata[:,1],ddc.ValueFn(ddc.ValueFn.xdata[:,1]));
end


begin "Check whether the optimal choice is correct"
    c_opt = deepcopy(ddc.PolicyFn.y);
    s = ddc.PolicyFn.xdata[:,1];
    η = ddc.PolicyFn.xdata[:,2];
    ϵ = minimum(s)*0.9;
    lb = ϵ;
    for n = 1:ddc.nSolve
        ub =  ddc.trans(0,s[n]) - ϵ;
        c_opt[n] = find_optim(s[n],η[n],lb,ub,ddc.β,ddc.trans,ddc.util,ddc.ValueFn);
    end
    s̃ = ddc.trans(c_opt,s);
    y=ddc.util(c_opt) + ddc.β * ddc.ValueFn(s̃);
    scatter(ddc.PolicyFn.xdata[:,1],c_opt);
    scatter!(ddc.PolicyFn.xdata[:,1],ddc.PolicyFn.y);
end

begin
    scatter(ddc.ValueFn.xdata[:,1],ddc.ValueFn.y);
    scatter!(ddc.ValueFn.xdata[:,1],value_function.(ddc.ValueFn.xdata[:,1]));
    scatter!(ddc.ValueFn.xdata[1:end-1,1],y);
end

begin
    c_opt = deepcopy(ddc.PolicyFn.y);
    s = ddc.PolicyFn.xdata[:,1];
    s̃ = ddc.trans(c_opt,s);
    y=ddc.util(c_opt) + ddc.β * ddc.ValueFn(s̃);
    scatter(ddc.ValueFn.xdata[:,1],ddc.ValueFn.y);
    scatter!(ddc.ValueFn.xdata[:,1],value_function.(ddc.ValueFn.xdata[:,1]));
    scatter!(ddc.ValueFn.xdata[1:end-1,1],y)
end

begin
    ee_scatter=[check_ee(ddc) for i = 1:100];
    scatter(ee_scatter);
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
    a_t = [];
    a_t1 = [];
    s_t = [];
    for t = 1:(nT-1)
        global a_t  = vcat(a_t,data.action[:,t]);
        global s_t  = vcat(s_t,data.state[:,t]);
        global a_t1 = vcat(a_t1,data.action[:,t+1])
    end
end


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
f_obj(result.minimizer)
f_obj((0.5,0.01))


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
