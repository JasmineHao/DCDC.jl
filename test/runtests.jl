using DCDC
using Test

# Multi-Variable Kernel Regression
@testset "Kernel Regressions" begin
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

    @test any(abs.(β_kernel - [1,2]) .< 1)
    @test any(abs.(β_OLS - [1,2]) .< 1)
end

# Single Variable Kernel Regression
@testset "Single Variable Kernel" begin
    n = 300
    xdata = randn(n,1);
    x = 0.3
    h = 0.1;
    y = xdata .+ 1 + randn(n);
    w = zeros(n)
    ekernel2(x,xdata,h,w,n) #Assign value to weight
    w_diag = diagm(0=>w);
    β_kernel = inv(xdata'*w_diag * xdata) * (xdata' * w_diag * y)
    @test (abs.(β_kernel .- 3.3) .< 1)[1]
end
