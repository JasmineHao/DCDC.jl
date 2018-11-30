using LinearAlgebra, DataFrames,Optim, ForwardDiff, BenchmarkTools,Distributions,Expectations, QuantEcon, Statistics
using Distributions: invsqrt2π, log2π, sqrt2, invsqrt2
using DCDC
using Test
using Distributed, Suppressor

# cd(joinpath(DEPOT_PATH[1],"dev","DCDC"))
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
    @test β_kernel != β_OLS;
    @test any(abs.(β_OLS - [1,2]) .< 1)
end


@testset "Kernel bandwidth constant" begin
    @test abs(bw_constant(1,2,:epan) - 2.34) < 0.01;
    @test abs(bw_constant(1,4,:epan) - 3.03) < 0.01;
    @test abs(bw_constant(1,6,:epan) - 3.53) < 0.01;
    @test abs(bw_constant(1,4,:triw) - 3.72) < 0.01;
    @test abs(bw_constant(1,4,:gaussian) - 1.08) < 0.01;
end


param1 = Parameter();
param2 = Parameter([1,2],[1,2]);
@testset "Parameter test" begin
    @test param1.γ == [1]
    @test param2.γ == [1,2]
end
@testset "State" begin
    state = State()
    state = State(rand(),randn(3),rand())
    @test typeof(state) == State;
end
@testset "ProfitFn" begin
    pf = ProfitFn(param1);
    @test pf() == 0;
end

# Plot
using Plots

begin
    n=300;
    xdata=4*randn(n);
    y=sin.(xdata)+ 0.3*randn(n);
    scatter(xdata,y)
    af=ApproxFn(xdata,y,:gaussian,2);
    y_hat= af(xdata);
    scatter!(xdata,y_hat);
end


# DDC
@testset "GMM small test" begin
      n = 3000;
      sig = rand(3,3)
      mu = rand(3)
      mn = MvNormal(mu,sig * sig')
      x_m = rand(mn,n)';
      x = x_m[:,[1,2]]
      β₀ = [1,3];
      y = x * β₀ + x_m[:,3]
      @show mn.Σ.mat;
      @show cov(x);
      β_OLS  = inv(x'*x) *(x'*y)
      obj = θ->(begin
            η = (x * θ - y) ;
            η'*η
            end)
      result = optimize(obj,zeros(2),Newton(),autodiff = :forward)
      @test result.minimizer ≈ β_OLS ;
end

@testset begin "Test Optim"
      n = 3000;
      sig = [01.0 0.5 0.5;
             0.0 0.5 0.0 ;
             0.0 0.0 1.0];

      mu = [1,9,0];
      mn = MvNormal(mu,sig * sig')
      W = rand(mn,n)'
      x = W[:,1];z=W[:,2];ϵ=W[:,3];
      y = x * 3 + ϵ;
      β_OLS = inv(x'*x)*(x'*y);
      β_IV = inv(z'*x)*(z'*y);
      w = hcat(x,y,z);
      obj = θ->(begin
            η = (w[:,1] .* θ - w[:,2]) .* w[:,3];
            η'*η
            end)
      result = optimize(obj,zeros(1),Newton(),autodiff = :forward)
      # @show converged(result) || error("Failed to converge in $(iterations(result)) iterations")
      @show xmin = result.minimizer
      @show result.minimum
      @test (β_OLS - result.minimizer[1] )< 0.1
end


@testset "Dynamic Decision Process" begin 
    σ₀ = 1;
    β = 0.8;
    nM = 50;
    nT = 5;
    ddc = DynamicDecisionProcess(σ₀,0.8);
    UpdateVal!(ddc);
    plot(ddc.ValueFn.xdata,ddc.ValueFn.y);
    plot(ddc.PolicyFn.xdata,ddc.PolicyFn.y);
    data = simulate_ddc(nM,nT,ddc);
end


# Generate IV regression data
@testset "IV regression" begin
    n = 3000;
    sig = [1.0 0.5 0.5;
         0.0 0.5 0.0 ;
         0.0 0.0 1.0];

    mu = [1,9,0];
    mn = MvNormal(mu,sig * sig')
    W = rand(mn,n)'
    x = W[:,1];z=W[:,2];ϵ=W[:,3];
    y = x * 3 + ϵ;
    β_OLS = inv(x'*x)*(x'*y);
    β_IV = inv(z'*x)*(z'*y);
    w = hcat(x,y,z);
    @test β_OLS != β_IV;
end
