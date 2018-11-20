using Pkg
Pkg.activate(".")
# include("./src/DCDC.jl")
using DCDC
using Test
p = Param()  #True ⁠θ
s = State()

profit_func = ProfitFn(p)
profit_func(s,1)

nSolve   =400 #The grid size, to solve for appriximate function
nTime    = 5
nMarket  = 10
nFirm    = 1


# The current payoff


# The ApproxFn
#_______________________________________________________________
# Linear

# The DynamicDecisionProcess
ddc = DynamicDecisionProcess(1,0.95)

# Define utility
σ = 1; #Elasticity
util = Utility(σ);

nSolve = 400;
ϵ = 0.005;
β = 0.95;
x = hcat(range(2*ϵ,step= ϵ,length=nSolve)); #Convert it into 2dimension
y = (util(x)./(1 -β))[:,1];
vf = ApproxFn(x,y);

g_tran = x -> ForwardDiff(f,)

global iter = 0
tol = 0.2
v_diff = Inf
using Suppressor

while (iter < 50 && v_diff > tol)
    c_opt = zeros(nSolve);
    for n = 1:nSolve
        @suppress begin
            ff = c ->  - (util(c' * [1]) + β * vf(Transition(x[n],c' * [1]))' * [1]);
            f_opt = optimize(ff,[ϵ],[x[n] - ϵ],[ϵ],SAMIN(),Optim.Options(g_tol = 1e-12,
                             iterations = 15,
                             store_trace = false,
                             show_trace = false));
            c_opt[n] = f_opt.minimizer[1];
         end
    end

    y = util(c_opt) + β * vf(Transition.(x,c_opt));
    print(iter);
    @show global v_diff = maximum(abs.(y - vf.y));
    UpdateVal(vf,y);
    global iter += 1;
end

using Plots
plot(vf.x,vf.y)
plot(vf.x,vf.y)
