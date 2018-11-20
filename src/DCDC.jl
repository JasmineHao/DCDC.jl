module DCDC
using LinearAlgebra, DataFrames,Optim, ForwardDiff, BenchmarkTools
using ScikitLearn,JLD,PyCall
using ScikitLearn: fit!, predict
using Distributed
@sk_import kernel_ridge: KernelRidge

using Distributions,Expectations, QuantEcon
using Distributions: invsqrt2π, log2π, sqrt2, invsqrt2
import Distributions.Normal, Expectations, QuantEcon
export  Param, State, profit, ProfitFn

RealVector = Union{Array{Float64},Array{Real},Array{Int}}
include("kernel.jl")
include("DCC.jl")
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
        self = new(x,y,n,"KRR",KernelRidge());
        fit!(self.model,x,y);
        return(self)
    end

    function (self::ApproxFn)(x_in::Real)
        y_in = predict(self.model,hcat([x_in]))
        return(y_in)
    end

    function (self::ApproxFn)(x_in::RealVector)
        y_in = predict(self.model,x_in)
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
    return(self)
end

function UpdateVal(self::ApproxFn,y)
    self.y     = y;
    return(self)
end

function Transition(x::Real,c::Real)
    return(1.04 * ( x - c ) + 1 )
end

# How to use sub types?
mutable struct DynamicDecisionProcess
    σ::Float64
    u::Utility
    ValueFn::ApproxFn
    β::Float64
    function DynamicDecisionProcess(σ::Real,β::Float64)
        x = randn(400,1)
        y = zeros(400)
        vf = ApproxFn(x,y)
        new(float(σ),Utility(float(σ)),x->x,β)
    end
end

end # module]
