# include("kernel.jl")
# include("DCC.jl")
# From this line
# This function is: util(c,η,s) = (1+η) c^(1-σ)/(1-σ)
mutable struct Utility
    σ::Float64
    Utility(σ::Real) = new(Float64(σ))
    function (self::Utility)(c::Union{Real,RealVector})
        if (self.σ == 1)
            return(log.(c))
        else
            return(float.(c).^(1 - self.σ)./(1 - self.σ));
        end
    end
    function (self::Utility)(c::Union{Real,RealVector},η::Union{Real,RealVector})
        if (self.σ == 1)
            return((η.+1).*log.(c))
        else
            return((η.+1).* (float.(c).^(1 - self.σ))./(1 - self.σ));
        end
    end

end


function Transition(s::Real,c::Real)
    return(1.05 * ( s - c ) + 1 )
end

function Transition(s::RealVector,c::RealVector)
    return(1.05 * ( s - c ) .+ 1 )
end

# How to use sub types?
function find_optim(xin::Real,ηin::Real, lb::Real, β::Float64, trans::Function, util::Utility,ValueFn::ApproxFn)
    @suppress begin
        # η = 0.01;
        ff = c ->  - (util(c' * [1],ηin) + β * ValueFn(trans(xin,c' * [1])));
        f_opt = optimize(ff,[lb],[xin[1] - lb],[lb],SAMIN(),Optim.Options(g_tol = 1e-12,
                         iterations = 100,
                     store_trace = false,
                     show_trace = false));
        return(f_opt.minimizer[1])
    end
end

mutable struct DynamicDecisionProcess
    σ::Float64
    util::Utility
    trans::Function
    PolicyFn::ApproxFn
    ValueFn::ApproxFn
    β::Float64
    dtrans::Function
    dutil::Function
    nSolve::Int
    function DynamicDecisionProcess(σ::Real,β::Float64)
        nSolve = 500;
        v_tol = 0.001;
        ϵ = 0.005;
        util = Utility(σ);
        # x = hcat(range(2*ϵ,step= ϵ,length=nSolve)); #Convert it into 2dimension
        s = convert(Array{Float64,1},range(2*ϵ,step=ϵ,length=nSolve)); #Convert it into 2dimension
        # η = randn(nSolve)./3; #Normal distributed error
        η=zeros(nSolve);
        # v = (util(s)./(1 -β));
        sdat = hcat(s,η);


        c_opt = zeros(nSolve);
        PolicyFn = ApproxFn(sdat,c_opt,:gaussian,2);

        ValueFn = ApproxFn(s,c_opt,:gaussian,2);
        ValueFn.h = 4.5; #Try this
        dtrans = (s,c) -> Tracker.gradient(Transition,s,c);
        dutil  = (c,η) -> Tracker.gradient(util,c,η)[1].data;
        ddc = new(float(σ),util,Transition,PolicyFn,ValueFn,β,dtrans,dutil,nSolve);
        return(ddc);
    end
end

# dtrans = (x,c) -> Tracker.gradient(ddc.trans,x,c) #Transition derivatives, can be broadcasted
# Compute equilibrium(): similar to Paul Schrimpf's dynamic choice problem

# While normV + normS2 + normV2 > tol (What is tole)
    # UpateValue, updateStrategy     : Update the solved value for policy and value function
    # gridUpdate && abs2(strategy.val,stratGrid.val)>strategy.val.size()*0.2
    # # UpdateSolvedGrid: To make the distribution stationary
    # stratGrid = strategy

function computeDistance(ddc::DynamicDecisionProcess,old_policy::ApproxFn,old_value::ApproxFn)
     normPolicy = abs2(old_policy.y - ddc.PolicyFn.y)./ddc.PolicyFn.n;
     normValue  = abs2(old_value.y  - ddc.ValueFn.y)./ddc.ValueFn.n;
     normData   = mean(abs2(old_value.xdata - ddc.ValueFn.xdata));
     return(normPolicy+normValue+normData);
end

function UpdateSolvedGrid!(ddc::DynamicDecisionProcess,T)
    # Update the grade for solving the points
    # ddc.policy
    # η_new = randn(ddc.nSolve); #Generate η
    η_new = zeros(ddc.nSolve); #Generate η
    s_new = ddc.PolicyFn.xdata[:,1];
    x_new = hcat(s_new,η_new);
    s_old = s_new;
    t = 0;
    while t < T
        @show t;
        s_new = ddc.trans(s_old,ddc.PolicyFn(x_new));
        η_new = randn(ddc.nSolve)./3; #Generate η
        x_new = hcat(s_new,η_new);
        s_diff= abs2(s_old - s_new)/ddc.nSolve;
        s_old = deepcopy(s_new);
        t += 1;
    end
end

function UpdateVal!(ddc::DynamicDecisionProcess)
    c_opt = ddc.PolicyFn.y;
    s = ddc.PolicyFn.xdata[:,1];
    η = ddc.PolicyFn.xdata[:,2];
    iter = 0
    tol = 1
    v_diff = Inf
    while (iter < 3 && v_diff > tol)
        for n = 1:ddc.nSolve
            c_opt[n] = find_optim(s[n],η[n],1e-6,ddc.β,ddc.trans,ddc.util,ddc.ValueFn);
        end

        y = ddc.util(c_opt) + ddc.β * ddc.ValueFn(Transition(s,c_opt));
        # println("Iteration:",iter);
        # @show iter;
        v_diff = mean(abs.(y - ddc.ValueFn.y));
        UpdateVal(ddc.ValueFn,y);
        iter += 1;
    end
    UpdateVal(ddc.PolicyFn,c_opt);
end

function computeEquilibrium(ddc::DynamicDecisionProcess)
    i = 0;
    diff_v=1;
    while i < 30 & diff_v > 1e-3
        old_value=deepcopy(ddc.ValueFn); old_policy=deepcopy(ddc.PolicyFn);
        UpdateVal!(ddc);
        @show diff_v=computeDistance(ddc,old_policy,old_value);
        UpdateSolvedGrid!(ddc::DynamicDecisionProcess,3);
        i+=1;
    end
end
# simulate dynamic discrete choice problem from the solved problem
function simulate_ddc(nM,nT,ddc::DynamicDecisionProcess)
    lb = minimum(ddc.PolicyFn.xdata,dims=1);
    ub = maximum(ddc.PolicyFn.xdata,dims=1);
    if lb[2]==ub[2]
        ub[2] = lb[2]+1e-6
    end
    x0 = zeros(nM,ddc.PolicyFn.q); #Initial states
    for i = 1:ddc.PolicyFn.q
        du = Uniform(lb[i],ub[i]);
        x0[:,i] = rand(du,nM);
    end
    s = zeros(nM,nT+1);
    a = zeros(nM,nT);
    s[:,1] = x0[:,1];
    for t = 1:nT
        x = hcat(s[:,t],randn(nM)/3);
        a[:,t] = ddc.PolicyFn(x);
        s[:,t+1] = ddc.trans(s[:,t],a[:,t]);
    end
    return(state=s,action=a);
end

function check_ee(ddc::DynamicDecisionProcess)
    s0=3*rand();
    ϵ0=0;
    PolicyFn=ApproxFn(ddc.PolicyFn.xdata[:,1],ddc.PolicyFn.y,:gaussian,2)
    a0=PolicyFn([s0])[1];
    s1=ddc.trans(s0,a0);
    a1=PolicyFn([s1])[1];
    return(ddc.dutil(a0,0)-1.05*ddc.β*ddc.dutil(a1,0));
end
