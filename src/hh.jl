# include("kernel.jl")
# include("DCC.jl")
# From this line
# This function is
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
end


function Transition(x::Real,c::Real)
    return(1.05 * ( x - c ) + 1 )
end

function Transition(x::RealVector,c::RealVector)
    return(1.05 * ( x - c ) .+ 1 )
end

# How to use sub types?
function find_optim(xin::Real, ϵ::Real, Transition::Function, util::Utility,vf::ApproxFn)
    @suppress begin
        # ϵ = 0.01;
        ff = c ->  - (util(c' * [1]) + β * vf(Transition(xin,c' * [1])));
        f_opt = optimize(ff,[ϵ],[xin - ϵ],[ϵ],SAMIN(),Optim.Options(g_tol = 1e-12,
                         iterations = 15,
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
    function DynamicDecisionProcess(σ::Real,β::Float64)
        nSolve = 500;
        ϵ = 0.01;
        util = Utility(σ);
        # x = hcat(range(2*ϵ,step= ϵ,length=nSolve)); #Convert it into 2dimension
        x = convert(Array{Float64,1},range(2*ϵ,step= ϵ,length=nSolve)); #Convert it into 2dimension
        y = (util(x)./(1 -β));
        vf = ApproxFn(x,y,:gaussian,2);

        iter = 0
        tol = 1
        v_diff = Inf

        c_opt = zeros(nSolve);
        while (iter < 50 && v_diff > tol)
            for n = 1:nSolve
                c_opt[n] = find_optim(x[n],ϵ,Transition,util,vf);
            end

            y = util(c_opt) + β * vf(Transition(x,c_opt));
            println("Iteration:",iter);
            @show v_diff = maximum(abs.(y - vf.y));
            UpdateVal(vf,y);
            iter += 1;
        end
        # policy = xin -> find_optim(xin,ϵ,Transition,util,vf);
        policy = ApproxFn(x,c_opt,:gaussian,2);
        dtrans = (x,c) -> Tracker.gradient(ddc.trans,x,c);
        dutil = (x) -> Tracker.gradient(ddc.util,x);
        new(float(σ),util,Transition,policy,vf,β,dtrans,dutil);
    end
end

# dtrans = (x,c) -> Tracker.gradient(ddc.trans,x,c) #Transition derivatives, can be broadcasted

function simulate_ddc(nM,nT,ddc::DynamicDecisionProcess)
    x_data = randn();
    # du = DiscreteUniform(1,500);
    lb = minimum(ddc.PolicyFn.xdata,dims=1);
    ub = maximum(ddc.PolicyFn.xdata,dims=1);
    x0 = zeros(nM,ddc.PolicyFn.q); #Initial states
    for i = 1:ddc.PolicyFn.q
        du = Uniform(lb[i],ub[i]);
        x0[:,i] = rand(du,nM);
    end
    x = zeros(nM,nT+1);
    c = zeros(nM,nT);
    x[:,1] = x0;
    for t = 1:nT
        c[:,t] = ddc.PolicyFn(x[:,t]);
        x[:,t+1] = ddc.trans(x[:,t],c[:,t]);
    end
    return(state=x,action=c);
end
