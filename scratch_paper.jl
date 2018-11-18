# Good Example
#_______________________________________________________________
mutable struct Person
    name::AbstractString
    male::Bool
    age::Float64
    children::Int
    describe::Function
    function Person(name,male,age,children)
        this = new()
        this.name = name
        this.male = male
        this.age = age
        this.children = children
        # anonymous functions are not known to be fast ;-)
        this.describe =  function() describe(this) end
        this
    end
end


function describe(p::Person)
    println("Name: ", p.name, " Male: ", p.male)
    println("Age: ", p.age, " Children: ", p.children)
end

ted = Person("Ted",1,55,0)
# describe(ted)
ted.describe()


# Reference is dynamic
#_______________________________________________________________
# For the self:: you can put anything
foo()::Function = println("a")
mutable struct PF
    x::Real
    y::Real
    print::Function
    PF(x,y) = new(x,y,foo) #constructor of a profit function
    (obj::PF)() = println(obj.x)
    (obj::PF)(k) = println(k)
    # function profit(s,i) = profit(s,i,p)
end
f = PF(1,2)
f.print() #We can assign functions as field

#  Ordered pair
#_______________________________________________________________

struct OrderedPair
   x::Real
   y::Real
   OrderedPair(x,y) = x > y ? error("out of order") : new(x,y)
   (obj::OrderedPair)() = println(obj.x)
   function fs(a)
        1
    end
end

#  Polynomial Example
#_______________________________________________________________
struct Polynomial{R}
   coeffs::Vector{R}
end

function (p::Polynomial)(x)
   v = p.coeffs[end]
   for i = (length(p.coeffs)-1):-1:1
       v = v*x + p.coeffs[i]
   end
   return v
end
(p::Polynomial)() = p(5)
# Define polynomial
p = Polynomial([1,10,100])
p(3)


# Nonparametric estimation
#_______________________________________________________________
using Distributions, KernelEstimator
using Plots
# using GR
gr()
x = rand(Chisq(2), 1000)
xs = range(0.01, stop=10, length=500)
dentrue = pdf(Chisq(2), xs)
dengamma = kerneldensity(x, xeval=xs, kernel=gammakernel, lb=0.0)
dennormal = kerneldensity(x, xeval=xs)
dennormal2 = kerneldensity(x, xeval=xs, h=.3)
histogram(x, bins=:scott,normed=true)
plot!(xs,dentrue)
plot!(xs,dengamma)
plot!(xs,dennormal)
title!("TITLE")

# Alternative kernel
xdata = randn(1000)
kerneldensity(xdata)
xeval = range(-3, stop=3, length=100)
bw = bwlscv(xdata, gaussiankernel)
kerneldensity(xdata, xeval=xeval, lb=-Inf, ub=Inf, kernel=gaussiankernel,h = bw)

x = rand(Beta(4,2), 500) * 10
y=2 .* x.^2 + x .* rand(Normal(0, 5), 500)
y_hat1 = npr(x, y)
y_hat2 = npr(x, y, xeval=x, reg=locallinear, kernel=betakernel,lb=0.0, ub=10.0)
plot(x,y)
plot!(x,y_hat1)

# Nonparametric.jl Demo
# _______________________________________________________________
# Kernel density estimate
 using Distributions
 x=rand(Normal(), 500)
 xeval=range(minimum(x), stop=maximum(x),length=100)
 den=kerneldensity(x, xeval=xeval)
# Local regression
 y=2 .* x.^2 + rand(Normal(), 500)
 yfit0=localconstant(x, y, xeval=xeval)
 yfit1=locallinear(x, y, xeval=xeval)
 yfit0=npr(x, y, xeval=xeval, reg=localconstant)
 yfit1=npr(x, y, xeval=xeval, reg=locallinear)
# Confidence Band
 cb=bootstrapCB(x, y, xeval=xeval)
 using Gadfly
 # plot(layer(x=x, y=y, Geom.point), layer(x=xeval, y=yfit1, Geom.line),
   # layer(x = xeval, y = cb[1,:], Geom.line),
   # layer(x=xeval, y=cb[2,:], Geom.line))

plot(x,y,line=(5,:dot,:green))
plot!(xeval,yfit0,line=(5,:dot,:red))
plot!(xeval,yfit1,line=(5,:dot,:orange))
