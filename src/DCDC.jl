module DCDC
    using LinearAlgebra, DataFrames,Optim, ForwardDiff, BenchmarkTools,Distributions,Expectations, QuantEcon
    using Distributions: invsqrt2π, log2π, sqrt2, invsqrt2
    # using ScikitLearn,JLD,PyCall
    # @sk_import kernel_ridge: KernelRidge
    # @sk_import svm: SVR
    # @sk_import neighbors: KNeighborsRegressor
    # using ScikitLearn: fit!, predict
    using Distributed, Suppressor

    # export Utility,ApproxFn,UpdateVal,UpdateData, DynamicDecisionProcess, Transition
    export ekernel4, ekernel2

    RealVector = Union{Array{Float64},Array{Real},Array{Int}}
    include("kernel.jl")
# include("DCC.jl")
# include("household.jl")
end # module]
