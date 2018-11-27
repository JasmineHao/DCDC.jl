using LinearAlgebra, DataFrames,Optim, ForwardDiff, BenchmarkTools,Distributions,Expectations, QuantEcon, Statistics
using Distributions: invsqrt2π, log2π, sqrt2, invsqrt2
using DCDC
using Test
using Distributed, Suppressor

begin
    σ=1;
    β=0.8;
    ddc = DynamicDecisionProcess(σ,β);
    plot(ddc.ValueFn.xdata,ddc.ValueFn.y);
    plot!(ddc.PolicyFn.xdata,ddc.PolicyFn.y)
end

nM = 50;
nT = 5;

function simulate_ddc(nM,nT,ddc::DynamicDecisionProcess)
    x_data = randn();
    # du = DiscreteUniform(1,500);
    lb = minimum(ddc.PolicyFn.xdata,dims=1);
    ub = maximum(ddc.PolicyFn.xdata,dims=1);
    x0 = zeros(nM,ddc.PolicyFn.q); #Initial states
    for i = 1:ddc.PolicyFn.q
        du = Uniform(lb[i],ub[i]);
        x0[:,i] = rand(du,nM);
    end
    x = zeros(nM,nT);
    x[:,1] = x0;
    for t = 1:nT
    end
end
