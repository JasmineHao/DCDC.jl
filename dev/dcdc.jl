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
# u'(a_t) = β d_Trans(a_t,x_t) u'(a_t+1)
function convert_data(data)
    a_t  = [];
    a_t1 = [];
    s_t  = [];
    s_t1 = [];
    for t = 1:(nT-1)
        a_t  = vcat(a_t,data.action[:,t]);
        s_t  = vcat(s_t,data.state[:,t]);
        a_t1 = vcat(a_t1,data.action[:,t+1]);
        s_t1 = vcat(s_t1,data.action[:,t+1]);
    end
    return(action_t=a_t,action_t_1=a_t1,state_t=s_t,state_t_1=s_t1);
end

reshaped_data = convert_data(data);

using Flux #Get derivatives
using Flux.Tracker
using Flux.Tracker: update!


θ₀ = ones(2)  #θ = (β,σ)
Transition = ddc.trans;
dtrans = (x,c) -> Tracker.gradient(Transition,x,c);
dtransx = (x,c) -> Tracker.gradient(Transition,x,c)[1];
R = dtransx.(reshaped_data.action_t,reshaped_data.state_t);
θ = θ₀

lb = [0,0];
ub = [1,100];
function _moment(θ)
    (β₀,σ₀ ) = θ₀;
    util = Utility(σ₀);
    dutil = (x) -> Tracker.gradient(util,x)[1].data;
    dudc  = dutil.(reshaped_data.action_t);
    dudc1 = dutil.(reshaped_data.action_t_1);
    η = dudc - β₀ .* dudc1 .* R;
    return(η'*η)
end

function _moment2(σ₀)
    dutil = (x) -> Tracker.gradient(Utility(σ₀);,x)[1].data;
    dudc  = dutil.(reshaped_data.action_t);
    dudc1 = dutil.(reshaped_data.action_t_1);
    η = dudc - 0.8 .* dudc1 .* R;
    return(η'*η)
end

optimize(_moment,zeros(2),BFGS(),autodiff=:forward)

function check_ee(ddc::DynamicDecisionProcess)
    
end
