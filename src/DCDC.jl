module DCDC
using Distributions,Expectations
import Distributions.Normal, Expectations
export greet,foo, Param, State, profit, ProfitFn
greet() = print("Hello World!")
function foo(μ = 1., σ = 2.)
    d = Normal(μ, σ)
    E = expectation(d)
    return E(x -> sin(x))
end



# Π(θ) = γ[1] * quality + γ[2] log(y - p)
# C(i;θ) = (β[1] - β[2] * η ) i + β[3] i^2 + β[4] ownQ 1(i>0& ownQ > 0 )

struct Param
    γ::Array{Real,1}
    β::Array{Real,1}
    Param() = new([1],[0.2, 0.1, 0.2, 0.0])
    Param(γ,β) = new(γ,β)
end
#
struct State
    ownQ::Real
    otherQ::Array{Real,1}
    η::Real  #The private shock
    State() = new(sqrt(3) * randn() + 0.5,[],sqrt(2)*randn());
    State(ownQ::Real,otherQ::Array{Real,1},η::Real) = new(ownQ,otherQ,η)
end
#
function profit(s::State,invest::Real,p::Param)
    profit = s.ownQ * p.γ[1]
    cost = (invest * (p.β[1] - p.β[2] * s.η) +
    p.β[3] * invest^2 + p.β[4]*(invest>0) * (s.ownQ > 0) * invest * s.ownQ)
    return profit - cost
end
#
mutable struct ProfitFn
    γ::Array{Real,1}
    β::Array{Real,1}
    p::Param
    ProfitFn(p) = new(p.γ, p.β,p) #constructor of a profit function
    (self::ProfitFn)() = 0
    (self::ProfitFn)(k) = println(k)
    (self::ProfitFn)(s,i) = profit(s,i,self.p) #callable struct
    # function profit(s,i) = profit(s,i,p)
end


# mutable struct name
#     fields
# end

end # module]
