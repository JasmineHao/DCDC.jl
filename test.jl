using Pkg
Pkg.activate(".")
# include("./src/DCDC.jl")
using DCDC
# include("./src/MyModule.jl")

using DCDC
# using DCDC
using Test
p = Param()  #True ⁠θ
s = State()

profit_func = ProfitFn(p)
profit_func(s,1)

nSolve   =400 #The grid size, to solve for appriximate function
nTime    = 5
nMarket  = 10
nFirm    = 1
