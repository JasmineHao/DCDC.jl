using DCDC
using Test

# Single type of kernel
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

@testset "Kernel Regressions" begin
    @test (β_kernel - [1,2]) .< 1
    @test (β_OLS - [1,2]) .< 1
end

# Plots
n = 300
xdata = randn(n,1);
x = 0.3
h = 0.1;
y = xdata .+ 1 + randn(n);
w = zeros(n)
ekernel2(x,xdata,h,w,n) #Assign value to weight
w_diag = diagm(0=>w);
β_kernel = inv(xdata'*w_diag * xdata) * (xdata' * w_diag * y)
