module DCDC
    using LinearAlgebra, DataFrames,Optim, ForwardDiff, BenchmarkTools,Distributions,Expectations, QuantEcon, Statistics
    using Distributions: invsqrt2π, log2π, sqrt2, invsqrt2
    # using ScikitLearn,JLD,PyCall
    # @sk_import kernel_ridge: KernelRidge
    # @sk_import svm: SVR
    # @sk_import neighbors: KNeighborsRegressor
    # using ScikitLearn: fit!, predict
    using Distributed, Suppressor

    # export Utility,ApproxFn,UpdateVal,UpdateData, DynamicDecisionProcess, Transition
    export RealVector,ekernel4, ekernel2, Kernel,estimate,
    forecast,bw_constant,compute_w,ApproxFn


    include("kernel.jl")
    include("param.jl")
    # include("hh.jl")
end # module]
