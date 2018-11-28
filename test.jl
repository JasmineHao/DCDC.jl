using LinearAlgebra, DataFrames,Optim, ForwardDiff, BenchmarkTools,Distributions,Expectations, QuantEcon, Statistics, GLM
using Distributions: invsqrt2π, log2π, sqrt2, invsqrt2
using DCDC
using Test
using Distributed, Suppressor


σ = 1;
β = 0.8;
nM = 50;
nT = 5;
ddc = DynamicDecisionProcess(σ,0.8);
plot(ddc.ValueFn.xdata,ddc.ValueFn.y);
plot!(ddc.PolicyFn.xdata,ddc.PolicyFn.y);
data = simulate_ddc(nM,nT,ddc);


begin "GLM"
    x = randn(300,3);
    y = x * [1,3,4] + randn(300);
    testdata = DataFrame(x);
    testdata.y = y;
    ols = lm(@formula(y ~ x1+x2+x3), testdata);
    @show stderror(ols);
    # @show ols.model.pp.beta0;
end

using ForwardDiff
begin "ForwardDiff"
    h(x) = sin(x[1]) + x[1] * x[2] + sinh(x[1] * x[2]) # multivariate.
    x = [1.4 2.2]
    @show ForwardDiff.gradient(h,x) # use AD, seeds from x
    #Or, can use complicated functions of many variables
    f(x) = sum(sin, x) + prod(tan, x) * sum(sqrt, x)
    g = (x) -> ForwardDiff.gradient(f, x); # g() is now the gradient
    @show g(rand(20)); # gradient at a random point

    function squareroot(x) #pretending we don't know sqrt()
        z = copy(x) # Initial starting point for Newton’s method
        while abs(z*z - x) > 1e-13
            z = z - (z*z-x)/(2z)
        end
        return z
    end
    sqrt(2.0)
    dsqrt(x) = ForwardDiff.derivative(squareroot, x)
    dsqrt(2.0)
end


begin "Multi-variate function: Can we forwarddiff?"
    func = (x,y) -> x^3 + y^2;
    kk  = y->(x->func(x,y));
    a,b = randn(2);
    ans1 = kk(b)(a);
    ans2 = func(a,b);
    @show ans1 == ans2;
    dfunx = (x,y) -> ForwardDiff.derivative(z->kk(y)(z),x);
    dfunx2 = (x,y) -> ForwardDiff.derivative(z->func(z,y),x);
    deriv1 =dfunx(a,b);
    deriv2 = dfunx2(a,b);
    @show deriv1 == deriv2;
end

begin "Test transition derivative"
    dtrans_x = (x,y) -> ForwardDiff.derivative( z-> ddc.trans(z,y),x) #Derivative w.r.t first component
    dtrans_c = (x,y) -> ForwardDiff.derivative( z-> ddc.trans(y,z),x) #Derivative w.r.t second component
    dtrans_x(1,3) == 1.05
end

using Flux
using Flux.Tracker
using Flux.Tracker: update!
begin "Test flux: graident and jacobian"
    f(x) = 3x^2 + 2x + 1
    # df/dx = 6x + 2
    df(x) = Tracker.gradient(f, x)[1]
    df(2);
    A = rand(2,2);
    f(x) = A * x
    x0 = [0.1, 2.0]
    f(x0)
    Flux.jacobian(f, x0)
end
