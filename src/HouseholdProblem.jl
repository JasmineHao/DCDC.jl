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
            return(exp.(η).*log.(c))
        else
            return(exp.(η).* (float.(c).^(1 - self.σ))./(1 - self.σ));
        end
    end

end

mutable struct Transition
    α::Real
    Transition(α)=new(α);
    function (self::Transition)(c::Union{Real,RealVector},s::Union{Real,RealVector})
    # return(1.1 * s.^(self.α) .- c)
    return(α*(s-c.+ 2.0))
    end
end

# function Transition(s::Real,c::Real)
    # return(1.05 * ( s - c ) + 3 )
# end

# function Transition(s::RealVector,c::RealVector)
    # return(1.05 * ( s - c ) .+ 3 )
# end

# How to use sub types?
function find_optim(xin::Real,ηin::Real, lb::Real, β::Float64, trans::Transition, util::Utility,ValueFn::ApproxFn)
    @suppress begin
        # η = 0.01;
        ff = c ->  - (util(c' * [1],ηin) + β * ValueFn(trans(c' * [1],xin)));
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
    trans::Transition
    PolicyFn::ApproxFn
    ValueFn::ApproxFn
    β::Float64
    dtrans::Function
    dutil::Function
    nSolve::Int
    function DynamicDecisionProcess(σ::Real,β::Float64,α::Real)
        nSolve = 100;
        v_tol = 0.001;
        ϵ     = 0.005;
        step_length= 2/nSolve;
        util  = Utility(σ);
        trans = Transition(α);
        # x = hcat(range(2*ϵ,step= ϵ,length=nSolve)); #Convert it into 2dimension
        s = convert(Array{Float64,1},range(2*ϵ,step=step_length,length=nSolve));
        # TN=TruncatedNormal(0, 1, 0, 10);
        # s = rand(TN,nSolve);
        #Convert it into 2dimension
        η = randn(nSolve); #Normal distributed error
        # η=zeros(nSolve);
        # v = (util(s)./(1 -β));
        sdat = hcat(s,η);


        c_opt = zeros(nSolve);
        PolicyFn = ApproxFn(sdat,c_opt,:gaussian,2);

        ValueFn = ApproxFn(s,c_opt,:gaussian,2);
        ValueFn.h = 5; #Try this
        dtrans = (c,s) -> Tracker.gradient(trans,c,s);
        dutil  = (c,η) -> Tracker.gradient(util,c,η)[1].data;
        ddc = new(float(σ),util,trans,PolicyFn,ValueFn,β,dtrans,dutil,nSolve);
        return(ddc);
    end
end

# dtrans = (c,s) -> Tracker.gradient(ddc.trans,c,s) #Transition derivatives, can be broadcasted
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
        s_new = ddc.trans(ddc.PolicyFn(x_new),s_old);
        η_new = randn(ddc.nSolve)/100; #Generate η
        x_new = hcat(s_new,η_new);
        s_diff= abs2(s_old - s_new)/ddc.nSolve;
        s_old = deepcopy(s_new);
        t += 1;
    end
end


function UpdateVal!(ddc::DynamicDecisionProcess,T::Int)
    c_opt = ddc.PolicyFn.y;
    s = ddc.ValueFn.xdata;
    η = ddc.PolicyFn.xdata[:,2];
    iter = 0
    tol = 1e-3
    v_diff = Inf
    while (iter < T && v_diff > tol)
        for n = 1:ddc.nSolve
            c_opt[n] = find_optim(s[n],η[n],1e-6,ddc.β,ddc.trans,ddc.util,ddc.ValueFn);
        end

        y = ddc.util(c_opt) + ddc.β * ddc.ValueFn(ddc.trans(s,c_opt));
        # println("Iteration:",iter);
        # @show iter;
        # @show v_diff = mean(abs.(y - ddc.ValueFn.y));
        UpdateVal(ddc.ValueFn,y);
        iter += 1;
    end
    UpdateVal(ddc.PolicyFn,c_opt);
end

function computeEquilibrium(ddc::DynamicDecisionProcess)
    i = 0;
    diff_v=Inf;
    tol = 1/ddc.nSolve;
    while (i < 30) & (diff_v > tol)
        old_value=deepcopy(ddc.ValueFn); old_policy=deepcopy(ddc.PolicyFn);
        UpdateVal!(ddc,3);
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
    du = Uniform(lb[2],ub[2]);
    for t = 1:nT
        # x = hcat(s[:,t],randn(nM)/3);
        x = hcat(s[:,t],rand(du,nM)); #This is a bit problematic
        a[:,t] = ddc.PolicyFn(x);
        s[:,t+1] = ddc.trans(a[:,t],s[:,t]);
    end
    return(state=s,action=a);
end

function check_ee(ddc::DynamicDecisionProcess)
    s0=rand();
    ϵ0=0;
    a0=ddc.PolicyFn([s0,ϵ0])[1];
    s1=ddc.trans(a0,s0);
    a1=ddc.PolicyFn([s1,0])[1];
    return(ddc.dutil(a0,0)- 1.05*ddc.β * ddc.dutil(a1,0));
end


mutable struct EulerEquation
    σ::Float64
    util::Utility
    trans::Transition
    PolicyFn::ApproxFn
    β::Float64
    dtrans::Function
    dutil::Function
    nSolve::Int
    function EulerEquation(σ::Real,β::Float64,α::Real)
        v_tol = 0.001;
        ϵ     = 0.005;
        util  = Utility(σ);
        trans = Transition(α);
        nSolve = 500;
        # x = hcat(range(2*ϵ,step= ϵ,length=nSolve)); #Convert it into 2dimension
        s = convert(Array{Float64,1},range(2*ϵ+1,step=ϵ,length=nSolve)); #Convert it into 2dimension
        η = randn(nSolve)/100; #Normal distributed error
        # η=zeros(nSolve);
        # v = (util(s)./(1 -β));
        sdat = hcat(s,η);
        c_opt = zeros(nSolve).+ϵ;
        tol=1e-3; diff=Inf;max_iter=10;iter=0;
        dtrans = (c,s) -> Tracker.gradient(trans,c,s);
        dutil  = (c) -> Tracker.gradient(util,c,0)[1].data;

        PolicyFn = ApproxFn(sdat,c_opt,:gaussian,2);
        dtrans_s = (c,s) -> dtrans(c,s)[2].data;

        function f_obj(c)
            diff_val =
            dutil.(c) - β.* dtrans_s.(c,PolicyFn.xdata[:,1]) .* PolicyFn( hcat(trans.(c,PolicyFn.xdata[:,1]),zeros(nSolve)) );
            return(sum(diff_val.^2))
        end

        while (diff > tol) & (iter<max_iter)

            result = optimize(f_obj,c_opt,PolicyFn.xdata[:,1],zeros(nSolve).+ϵ,SAMIN());
            # c_opt=result.minimizer;;
            @show diff = sum( (result.minimizer-PolicyFn.y).^2);
            UpdateVal(PolicyFn,result.minimizer);
            iter+=1;
        end
        ee = new(float(σ),util,trans,PolicyFn,β,dtrans,dutil,nSolve);
        return(ee);
    end
end


function check_ee(ee::EulerEquation)
    s0=rand();
    ϵ0=0;
    a0=ee.PolicyFn([s0,ϵ0])[1];
    s1=ee.trans(a0,s0);
    a1=ee.PolicyFn([s1,0])[1];
    return(ee.dutil(a0)- 1.05*ee.β * ee.dutil(a1));
end

function simulate_ee(nM,nT,ee::EulerEquation)
    lb = minimum(ee.PolicyFn.xdata,dims=1);
    ub = maximum(ee.PolicyFn.xdata,dims=1);
    if lb[2]==ub[2]
        ub[2] = lb[2]+1e-6
    end
    x0 = zeros(nM,ee.PolicyFn.q); #Initial states
    for i = 1:ee.PolicyFn.q
        du = Uniform(lb[i],ub[i]);
        x0[:,i] = rand(du,nM);
    end
    s = zeros(nM,nT+1);
    a = zeros(nM,nT);
    s[:,1] = x0[:,1];
    for t = 1:nT
        # x = hcat(s[:,t],randn(nM)/3);
        x = hcat(s[:,t],zeros(nM));
        a[:,t] = ee.PolicyFn(x);
        s[:,t+1] = ee.trans(a[:,t],s[:,t]);
    end
    return(state=s,action=a);
end
