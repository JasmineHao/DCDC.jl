using DCDC
using Test

@test foo() == 0.11388071406436832
@test foo(1, 1.5) == 0.2731856314283442
@test_broken foo(1, 0) # tells us this is broken
#@testset "DCDC.jl" begin
    # Write your own tests here.
#end
greet()
DCDC.foo()

# Profit function
p = Param()  #True ⁠θ

Param([3.0],[1.0])
profit_func = ProfitFn(p)
profit_func()
profit_func(310)

s = State()
i = 30.0
@test profit(s,i,p)==profit_func(s,i) #Define callable struct
