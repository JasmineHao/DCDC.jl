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
    A::Real
    Transition(α,A)=new(α,A);
    function (self::Transition)(c::Union{Real,RealVector},s::Union{Real,RealVector})
    # return(1.1 * s.^(self.α) .- c)
    # return(α*(s-c.+ 2.0))
    return (self.A * (s.^(self.α)) .- c)
    end
end

# function Transition(s::Real,c::Real)
    # return(1.05 * ( s - c ) + 3 )
# end

# function Transition(s::RealVector,c::RealVector)
    # return(1.05 * ( s - c ) .+ 3 )
# end

# How to use sub types?
function find_optim(xin::Real,ηin::Real, lb::Real,ub::Real, β::Float64, trans::Transition, util::Utility,ValueFn::ApproxFn)
    @suppress begin
        # η = 0.01;
        ff = c ->  - (util(c' * [1],ηin) + β * ValueFn(trans(c' * [1],xin)));
        # ub = trans(0,xin[1])-lb;
        @show init_val = (ub + lb) / 2;
        f_opt = optimize(ff,[lb],[ub],[lb],SAMIN(),Optim.Options(g_tol = 1e-12,
                         iterations = 1000,
                     store_trace = false,
                     show_trace = false));
        return(f_opt.minimizer[1])
    end
end

mutable struct State
    s::Union{Real,RealVector} #Observed
    η::Real #Unobserved
    State()=new(0,0);
    function State(s::Real, η::Real)
        new([s],η);
    end

    function State(s::Array, η::Real)
        new(s,η);
    end
end

function vectorize(s::State)
    return(vcat(s.s,s.η))
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
    function DynamicDecisionProcess(σ::Real,β::Float64,α::Real,A::Real)
        nSolve = 400;
        v_tol = 0.001;
        ϵ     = 0.05;
        # step_length= 2/nSolve;
        step_length=ϵ;
        util  = Utility(σ);
        trans = Transition(α,A);
        # x = hcat(range(2*ϵ,step= ϵ,length=nSolve)); #Convert it into 2dimension
        s = convert(Array{Float64,1},range(ϵ,step=step_length,length=nSolve));
        # TN=TruncatedNormal(0, 1, 0, 10);
        # s = rand(TN,nSolve);
        #Convert it into 2dimension
        η = randn(nSolve)/100; #Normal distributed error
        # η=zeros(nSolve);
        # v = (util(s)./(1 -β));
        sdat = hcat(s,η);

        s = vcat([1e-10*ϵ],s);
        c_opt = zeros(nSolve);
        v_first=zeros(nSolve+1);
        PolicyFn = ApproxFn(sdat,c_opt,:gaussian,2);
        ValueFn = ApproxFn(s,v_first,:gaussian,2);
        # PolicyFn.h =  0.5 * PolicyFn.h;
        # ValueFn.h =  0.5 * ValueFn.h ;
        # ValueFn.h = 1; #Try this
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
    η_new = randn(ddc.nSolve)/100; #Generate η
    # η_new = zeros(ddc.nSolve); #Generate η
    s_new = ddc.ValueFn.xdata;
    x_new = hcat(s_new,η_new);
    s_old = s_new;
    val_new = val_old = ddc.ValueFn.y;
    t = 0;
    while t < T
        # @show t;
        c_new = ddc.PolicyFn(x_new);
        s_new = ddc.trans(c_new,s_old);
        UpdateData(ddc.ValueFn,s_old,val_old);

        val_new = ddc.util(c_new,x_new[:,2]) + ddc.β * ddc.ValueFn(s_new);
        val_old = deepcopy(val_new);
        s_new[s_new.<0].=minimum(s_old);
        η_new = randn(ddc.nSolve); #Generate η
        x_new = hcat(s_new,η_new);
        s_diff= abs2(s_old - s_new)/ddc.nSolve;
        s_old = deepcopy(s_new);
        t += 1;
    end

end


function UpdateVal!(ddc::DynamicDecisionProcess,T::Int)
    c_opt = ddc.PolicyFn.y;
    s = ddc.PolicyFn.xdata[:,1];
    η = ddc.PolicyFn.xdata[:,2];
    iter = 0
    tol = 1e-8
    v_diff = Inf
    while (iter < T && v_diff > tol)
        ϵ = minimum(s)*0.9;
        lb = ϵ;
        for n = 1:ddc.nSolve
            ub = max(0.9 * ddc.trans(0,s[n]),ddc.trans(0,s[n])-2*ϵ);
            c_opt[n] = find_optim(s[n],η[n],lb,ub,ddc.β,ddc.trans,ddc.util,ddc.ValueFn);
        end

        y = ddc.util(c_opt) + ddc.β * ddc.ValueFn(ddc.trans(c_opt,s));
        # println("Iteration:",iter);
        # @show iter;
        # @show v_diff = mean(abs.(y - ddc.ValueFn.y));
        UpdateVal(ddc.ValueFn,vcat([ddc.ValueFn.y[1]],y));
        @show iter += 1;
    end
    UpdateVal(ddc.PolicyFn,c_opt);
end

function computeEquilibrium(ddc::DynamicDecisionProcess)
    i = 0;
    diff_v=Inf;
    diff_v_old = 0;
    diff_r=Inf;
    # tol = 1/ddc.nSolve;
    tol_r = 0.1;
    tol_v = 1e-5;
    # while (i < 30) & (diff_v > tol)
    while (i < 30) & (diff_r > tol_r) & (diff_v > tol_v)
        old_value=deepcopy(ddc.ValueFn); old_policy=deepcopy(ddc.PolicyFn);
        UpdateVal!(ddc,10);
        @show diff_v=computeDistance(ddc,old_policy,old_value);
        @show diff_r = abs(diff_v-diff_v_old) / diff_v;
        diff_v_old = diff_v;
        # UpdateSolvedGrid!(ddc::DynamicDecisionProcess,2);
        i+=1;
    end
end


# simulate dynamic discrete choice problem from the solved problem
function simulate_ddc(nM,nT,ddc::DynamicDecisionProcess)
    # lb = minimum(ddc.PolicyFn.xdata,dims=1);
    # ub = maximum(ddc.PolicyFn.xdata,dims=1);
    # if lb[2]==ub[2]
    #     ub[2] = lb[2]+1e-6
    # end
    # x0 = zeros(nM,ddc.PolicyFn.q); #Initial states
    # for i = 1:ddc.PolicyFn.q
    #     du = Uniform(lb[i],ub[i]);
    #     x0[:,i] = rand(du,nM);
    # end
    # du=DiscreteUniform(1,ddc.nSolve); #Draw from the nSolved states
    # s = zeros(nM,nT+1);
    # a = zeros(nM,nT);
    # s[:,1] = x0[:,1];
    # du = Uniform(lb[2],ub[2]);

    # du=DiscreteUniform(1,ddc.nSolve); #Draw from the nSolved states
    du=Uniform(2,20);
    s = [ State() for i=zeros(nM,nT)]; #nM x nT state
    a = [ 0.0 for i=zeros(nM,nT)]; #nM x nT action
    for n = 1:nM
        ind_1=rand(du);
        # s[n,1] = State(ddc.PolicyFn.xdata[ind_1,1:end-1],randn()/100);
        s[n,1] = State(rand(du),randn()/100);
        for t = 1:(nT-1)
        # x = hcat(s[:,t],randn(nM)/3);
            x = vectorize(s[n,t]); #This is a bit problematic
            a[n,t] = ddc.PolicyFn(x);
            s_1 = ddc.trans(a[n,t],s[n,t].s);
            if s_1[1] < 0
                @show n,t
                break
            end
            s[n,t+1] = State(s_1,randn()/100);

        end
        x = vectorize(s[n,nT]); #T
        a[n,nT] = ddc.PolicyFn(x);
    end

    return(state=s,action=a);
end

function check_ee(ddc::DynamicDecisionProcess)
    s0=rand(Uniform(3,15));
    ϵ0=0;
    a0=ddc.PolicyFn([s0,ϵ0])[1];
    s1=ddc.trans(a0,s0);
    a1=ddc.PolicyFn([s1,0])[1];
    ds=ddc.dtrans(a1,s1)[2].data;
    return(ddc.dutil(a0,0)- ds *ddc.β  * ddc.dutil(a1,0));
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
