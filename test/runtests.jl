using LinearAlgebra, DataFrames,Optim, ForwardDiff, BenchmarkTools,Distributions,Expectations, QuantEcon, Statistics
using Distributions: invsqrt2π, log2π, sqrt2, invsqrt2
using DCDC
using Test

begin
    n = 300
    xdata = randn(n,2);
    x = [0.3,0.4]
    h = [2,2];
    y = xdata * [1,2] + randn(n);
    w = zeros(n);
    ekernel4(x,xdata,h,w,n); #Assign value to weight
    w_diag = diagm(0=>w);
    β_kernel = inv(xdata'*w_diag * xdata) * (xdata' * w_diag * y)
    β_OLS = inv(xdata'*xdata) * (xdata'*y);

    af = ApproxFn(xdata,y,:gaussian,2);

end
# Multi-Variable Kernel Regression
@testset "Kernel Regressions" begin
    @test any(abs.(β_kernel - [1,2]) .< 1)
    @test any(abs.(β_OLS - [1,2]) .< 1)
end


@testset "Kernel bandwidth constant" begin
    @test abs(bw_constant(1,2,:epan) - 2.34) < 0.01;
    @test abs(bw_constant(1,4,:epan) - 3.03) < 0.01;
    @test abs(bw_constant(1,6,:epan) - 3.53) < 0.01;
    @test abs(bw_constant(1,4,:triw) - 3.72) < 0.01;
    @test abs(bw_constant(1,4,:gaussian) - 1.08) < 0.01;
end

n=300;
xdata=randn(n);
y=xdata*3+randn(n);
using Plots
scatter(xdata,y)
af=ApproxFn(xdata,y,:gaussian,2);
y_hat= af(xdata);
scatter!(xdata,y_hat);
