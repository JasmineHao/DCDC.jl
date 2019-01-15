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

#_____________________________________________________________________________
# linearize estimation method
#_____________________________________________________________________________

begin
    β_MC=zeros(10);
    for i = 1:10

        nM = 100;
        nT = 2;
        data = simulate_ddc(nM,nT,ddc);
        # data =simulate_ee(nM,nT,ee);
        # The moment condition is
        # u'(c_t) = β d_Trans(c_t,x_t) u'(c_t+1)
        a_t = []; a_t1 = [];
        s_t = []; s_t1 = [];
        for t = 1:(nT-1)
             a_t  = vcat(a_t,data.action[:,t]);
             s_t  = vcat(s_t, [s.s[1] for s in data.state[:,t]] );
             a_t1 = vcat(a_t1,data.action[:,t+1])
            s_t1  = vcat(s_t1, [s.s[1] for s in data.state[:,t+1]] );
        end
        a_t=convert(Array{Float64,1},a_t);
        s_t=convert(Array{Float64,1},s_t);
        a_t1=convert(Array{Float64,1},a_t1);
        s_t1=convert(Array{Float64,1},s_t1);
        dtrans=ddc.dtrans;
        dtrans_s = (a,s) -> dtrans(a,s)[2].data;
        R = dtrans_s.(a_t1,s_t1);
        y=log.(a_t1) - log.(a_t)
        X=hcat(ones(length(y)),log.(R));
        b=inv(X'*X)*X'*y;
        @show β_MC[i]=exp(b[1]/b[2]);
    end

    @show b=inv(X'*X)*X'*y;
    @show 1/b[2];
    @show "True beta"
    @show β
    @show exp(b[1]/b[2]);
    "End";
end

begin "The estimation"
    policy_approx = ApproxFn(s_t1,a_t1,:epan,2);
    scatter(s_t1,a_t1);
    scatter!(s_t1,policy_approx(s_t1));

end
# plot!(ee.PolicyFn.xdata[:,1],ee.PolicyFn.y);
# scatter(ddc.PolicyFn.xdata[:,2]);
# checked = [check_ee(ddc) for i = 1:100];
# scatter(checked[checked .< 100])
#

#_____________________________________________________________________________
# Nonlinearize estimation method
#_____________________________________________________________________________

using Flux #Get derivatives
using Flux.Tracker
using Flux.Tracker: update!
# Solve the Euler equations

lb = [0,0];
ub = [1,100];

σ̂=2;β̂=0.8;

begin
    nM = 100;
    nT = 3;
    data = simulate_ddc(nM,nT,ddc);
    # data =simulate_ee(nM,nT,ee);
    # The moment condition is
    # u'(c_t) = β d_Trans(c_t,x_t) u'(c_t+1)
    a_t = [];
    s_t = [];
    for t = 1:(nT)
         global a_t  = vcat(a_t,data.action[:,t]);
         global s_t  = vcat(s_t, [s.s[1] for s in data.state[:,t]] );
    end
    a_t=convert(Array{Float64,1},a_t);
    s_t=convert(Array{Float64,1},s_t);
end

trans=ddc.trans;
function _moment(θ,trans,a_t,s_t,policy_approx)
    σ̂,β̂=θ; s_t1_simul=trans(a_t,s_t);
    a_t1_simul=policy_approx(s_t1_simul);
    util=Utility(σ̂); dutil=(c,η) -> Tracker.gradient(util,c,η)[1].data;
    # η=randn(100)./100;
    dc_t=dutil.(a_t,zeros(length(a_t)));
    dc_t1=dutil.(a_t1_simul,zeros(length(a_t)));
    dtrans_s = (a,s) -> dtrans(a,s)[2].data;
    R = dtrans_s.(a_t1_simul,s_t1_simul);
    ϵ= (β̂ * dc_t1 .* R ) ./ dc_t .-1;
    ϵ.* s_t;
end


function f_obj(θ)
    CF =_moment(θ,ddc.trans,a_t,s_t,policy_approx);
    sum(CF)^2;
end

result = optimize(f_obj,[0.1,0.01])
result.minimizer
f_obj(result.minimizer)
f_obj([β,1]+randn(2))

# Results of Optimization Algorithm
#  * Algorithm: Nelder-Mead
#  * Starting Point: [0.1,0.01]
#  * Minimizer: [0.47126793573588627,1.0128364461302402]
#  * Minimum: 2.226862e-11
#  * Iterations: 49
#  * Convergence: true
#    *  √(Σ(yᵢ-ȳ)²)/n < 1.0e-08: true
#    * Reached Maximum Number of Iterations: false
#  * Objective Calls: 98
