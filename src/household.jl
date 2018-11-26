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


# The ApproxFn

mutable struct ApproxFn
    x::RealVector
    y::Array{Float64,1}
    n::Int
    method::String
    model::Any
    function ApproxFn(x::Array{Float64},y::Array{Float64,1})
        n = size(x,1);
        if length(y) != n
            error("The dimension of x,y must match");
        end
        # self = new(x,y,n,"KRR",KernelRidge());
        self = new(x,y,n,"SVR",SVR(kernel="rbf",degree=4,gamma="scale"));
        # self = new(x,y,n,"KNN",KNeighborsRegressor());
        fit!(self.model,x,y);
        return(self)
    end

    function (self::ApproxFn)(x_in::Real)
        y_in = predict(self.model,hcat([x_in]));
        return(y_in)
    end

    function (self::ApproxFn)(x_in::RealVector)
        y_in = predict(self.model,x_in);
        return(y_in)
    end

    # Tried the GMM
    # data = DataFrame(self.x);
    # ff = @formula(y ~ 1 + x)
    # ff.rhs.args = vcat([:+], names(data))
    # data[:y] = self.y;
    # ols = lm(ff,data)
    # xin_df = DataFrame(vcat(x_in)')
    # y_in = predict(ols,xin_df)

    # Tried KernelEstimator
    # w = zeros(self.n);
    # h = ones(size(self.x,1));
    # gaussiankernel(x_in, self.x, h , w, self.n);
    # return(w*y)
end


function UpdateData(self::ApproxFn,x,y)
    self.x     = x;
    self.y     = y;
    fit!(self.model,self.x,self.y);
    return(self)
end

function UpdateVal(self::ApproxFn,y)
    self.y     = y;
    fit!(self.model,self.x,self.y);
    return(self)
end

function Transition(x::Real,c::Real)
    return(1.05 * ( x - c ) + 1 )
end

# How to use sub types?
function find_optim(xin::Real, ϵ::Real, Transition::Function, util::Utility,vf::ApproxFn)
    @suppress begin
        ϵ = 0.01;
        ff = c ->  - (util(c' * [1]) + β * vf(Transition(xin,c' * [1]))' * [1]);
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
    policy::Function
    trans::Function
    ValueFn::ApproxFn
    β::Float64

    function DynamicDecisionProcess(σ::Real,β::Float64)
        nSolve = 500;
        ϵ = 0.01;
        util = Utility(σ);
        x = hcat(range(2*ϵ,step= ϵ,length=nSolve)); #Convert it into 2dimension
        y = (util(x)./(1 -β))[:,1];
        vf = ApproxFn(x,y);

        iter = 0
        tol = 1
        v_diff = Inf

        c_opt = zeros(nSolve);
        while (iter < 50 && v_diff > tol)
            for n = 1:nSolve
                c_opt[n] = find_optim(x[n],ϵ,Transition,util,vf);
            end

            y = util(c_opt) + β * vf(Transition.(x,c_opt));
            print(iter);
            @show v_diff = maximum(abs.(y - vf.y));
            UpdateVal(vf,y);
            iter += 1;
        end
        policy = xin -> find_optim(xin,ϵ,Transition,util,vf);
        new(float(σ),util,policy,Transition,vf,β);
    end
end

# function simulate(ddc::DynamicDecisionProcess)
#     x =
# end
