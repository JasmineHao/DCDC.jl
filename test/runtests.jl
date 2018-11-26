using LinearAlgebra, DataFrames,Optim, ForwardDiff, BenchmarkTools,Distributions,Expectations, QuantEcon
using Distributions: invsqrt2π, log2π, sqrt2, invsqrt2
using DCDC
using Test

begin
    n = 300
    xdata = randn(n,2);
    x = [0.3,0.4]
    h = [2,2];
    y = xdata * [1,2] + randn(n);
    w = zeros(n)
    ekernel4(x,xdata,h,w,n) #Assign value to weight
    w_diag = diagm(0=>w);
    β_kernel = inv(xdata'*w_diag * xdata) * (xdata' * w_diag * y)
    β_OLS = inv(xdata'*xdata) * (xdata'*y)
end
# Multi-Variable Kernel Regression
@testset "Kernel Regressions" begin

    @test any(abs.(β_kernel - [1,2]) .< 1)
    @test any(abs.(β_OLS - [1,2]) .< 1)
end

begin
    Kern = Kernel(xdata,y);
    for1 = forecast(x,Kern);
    for2 = Kern.forecast(x);
end
@testset "Test the integration of estimate function" begin
    @test for1 == for2;
    @test estimate(x,Kern) == Kern.estimate(x);
end
# Single Variable Kernel Regression
n = 300
xdata = randn(n,1);
x = 0.3
h = 1;
y = xdata + randn(n);
w = zeros(n)

@testset "Single Variable Kernel" begin
    ekernel4(x,xdata,h,w,n) #Assign value to weight
    w_diag = diagm(0=>w);
    β_kernel = inv(xdata'*w_diag * xdata) * (xdata' * w_diag * y)
    @test (abs.(β_kernel .- 1) .< 1)[1]
end
yfit = forecast_fit(Kern)
