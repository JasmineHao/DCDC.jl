module DCDC
    using LinearAlgebra, DataFrames,Optim, ForwardDiff, BenchmarkTools,Distributions,Expectations, QuantEcon, Statistics, GLM
    using Distributions: invsqrt2π, log2π, sqrt2, invsqrt2
    # using ScikitLearn,JLD,PyCall
    # @sk_import kernel_ridge: KernelRidge
    # @sk_import svm: SVR
    # @sk_import neighbors: KNeighborsRegressor
    # using ScikitLearn: fit!, predict
    using Distributed, Suppressor
    using Flux #Get derivatives
    using Flux.Tracker
    using Flux.Tracker: update!

    # export Utility,ApproxFn,UpdateVal,UpdateData, DynamicDecisionProcess, Transition
    export RealVector,ekernel4, ekernel2, Kernel,estimate,
    forecast,bw_constant,compute_w,ApproxFn,State,
    DynamicDecisionProcess,dynamic_decision_process,simulate_ddc,
    Utility, Transition, State, find_optim, UpdateVal!,computeEquilibrium,check_ee,
    convert_data, EulerEquation, simulate_ee, _moment
    # Parameter, State, profit, ProfitFn,

    include("kernel.jl")
    # include("param.jl")
    include("HouseholdProblem.jl")
    include("misc.jl")
end # module]

a = zeros(10)
Threads.@threads for i = 1:10
   a[i] = Threads.threadid()
end
