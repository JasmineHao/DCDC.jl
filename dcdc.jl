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
    ρ = 0.15;
    β = 1 / ( 1 + ρ);
    r = 0.2;
    α = 0.9;
    A = 1 + r;
    ddc = DynamicDecisionProcess(σ₀,β,α,A);
    # The log function has a solution
    value_function(s) = (log(1 - β * α) + (β * α) * log(β * α) /(1-β * α) ) / (1 - β) + α * log(s) /(1-β * α);
    policy_function(s) = (1 - β * α) * A * s^α;
    ddc.ValueFn.y = value_function.(ddc.ValueFn.xdata);
    ddc.PolicyFn.y = policy_function.(ddc.PolicyFn.xdata[:,1]);
    # computeEquilibrium(ddc);
    # ee = EulerEquation(σ₀,β,α);
end

# "Check Approximation of Policy Function"
begin
    scatter(ddc.PolicyFn.xdata[:,1],ddc.PolicyFn.y);
    scatter!(ddc.PolicyFn.xdata[:,1],ddc.PolicyFn(ddc.PolicyFn.xdata));
    scatter!(ddc.PolicyFn.xdata[:,1],policy_function.(ddc.PolicyFn.xdata[:,1]));
    # scatter!(ddc.PolicyFn.xdata[:,1],A * ddc.PolicyFn.xdata[:,1].^α)
end

# "Check Approximation of Value Function"
begin
    scatter(ddc.ValueFn.xdata[:,1] ,ddc.ValueFn.y);
    scatter!(ddc.ValueFn.xdata[:,1],ddc.ValueFn(ddc.ValueFn.xdata[:,1]));
end


begin "Check whether the optimal choice is correct"
    c_opt = deepcopy(ddc.PolicyFn.y);
    s = ddc.ValueFn.xdata;
    η = ddc.PolicyFn.xdata[:,2];
    ϵ = minimum(s)*0.9;
    s = ddc.PolicyFn.xdata[:,1];
    lb = ϵ;
    for n = 1:ddc.nSolve
        ub =  ddc.trans(0,s[n]) - ϵ;
        @show c_opt[n] = find_optim(s[n],η[n],lb,ub,ddc.β,ddc.trans,ddc.util,ddc.ValueFn);
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
    @show mean(ee_scatter)
    scatter(ee_scatter);
end

begin
    nM = 1000;
    nT = 2;
    data = simulate_ddc(nM,nT,ddc);
    # data =simulate_ee(nM,nT,ee);
    # The moment condition is
    # u'(c_t) = β d_Trans(c_t,x_t) u'(c_t+1)
    a_t = []; a_t1 = [];
    s_t = []; s_t1 = [];
    for t = 1:(nT-1)
        global a_t  = vcat(a_t,data.action[:,t]);
        global s_t  = vcat(s_t, [s.s[1] for s in data.state[:,t]] );
        global a_t1 = vcat(a_t1,data.action[:,t+1])
        global s_t1  = vcat(s_t1, [s.s[1] for s in data.state[:,t+1]] );
    end
    dtrans=ddc.dtrans;
    dtrans_s = (a,s) -> dtrans(a,s)[2].data;
    R = dtrans_s.(a_t1,s_t1);
    y=log.(a_t1) - log.(a_t)
    X=hcat(ones(length(y)),log.(R));
    @show b=inv(X'*X)*X'*y;
    @show 1/b[2];
    @show "True beta"
    @show β
    @show exp(b[1]/b[2]);
    "End";
end


# plot!(ee.PolicyFn.xdata[:,1],ee.PolicyFn.y);
# scatter(ddc.PolicyFn.xdata[:,2]);
# checked = [check_ee(ddc) for i = 1:100];
# scatter(checked[checked .< 100])
#

using Flux #Get derivatives
using Flux.Tracker
using Flux.Tracker: update!
# Solve the Euler equations

lb = [0,0];
ub = [1,100];


function f_obj(θ)
    ϵ=_moment(θ);
    return(ϵ' * ϵ);
end

result = optimize(f_obj,[0.0001,0.01],[0.999,10.0],[0.8,3.5])
result.minimizer
f_obj(result.minimizer)
f_obj((β,σ₀))


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
