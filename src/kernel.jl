import Distributions.estimate
rhoxb(x::Real, b::Real) = 2*b*b + 2.5 - sqrt(4*b^4 + 6*b*b+2.25 - x*x - x/b)
function multiply!(des::RealVector, x::RealVector, y::Real, n::Int=length(x))
    for i in 1:n
        @inbounds des[i] = x[i]*y
    end
end
multiply!(x::RealVector, y::Real) = multiply!(x, x, y)
function divide!(des::RealVector, x::RealVector, y::Real, n::Int=length(x))
    for i in 1:n
        @inbounds des[i] = x[i]/y
    end
end
divide!(x::RealVector, y::Real) = divide!(x, x, y)
function minus!(des::RealVector, y::Float64, x::RealVector, n::Int64=length(x))
   for i in 1:n
       @inbounds des[i] = y - x[i]
   end
   nothing
end
function add!(x::Vector{Float64}, y::Float64, n::Int64=length(x))
   for i in 1:n
       @inbounds x[i] = x[i] + y
   end
   nothing
end


function abs2!(des::RealVector, x::RealVector, n::Int64=length(x))
   for i in 1:n
       @inbounds des[i] = abs2(x[i])
   end
   nothing
end
import Base.abs2
function abs2(x::RealVector)
    return abs(x'*x)
end

# Second order gaussiankernel
function gaussiankernel(x::Real, xdata::RealVector, h::Real, w::Vector, n::Int)
    h1= 1.0/h
    tmp = log(h) + log2π/2
    for ind in 1:n
        @inbounds w[ind]=-0.5*abs2((x - xdata[ind])*h1) - tmp
    end
    w .= exp.(w)

    nothing
end

# MultiVariate prototype
function gaussiankernel(x::RealVector, xdata::RealVector, h::RealVector, w::Vector, n::Int)
    h1= 1.0/h
    tmp = log(prod(h)) + log2π/2
    x1 =x;
    xdata1 = xdata;
    for d = 1: size(h,1)
        @inbounds x1[d] = x[d] / h[d]
        @inbounds xdata1[:,d] = xdata[:,d] / h[d]
        d+=1
    end

    for ind in 1:n
        @inbounds w[ind]=-0.5*abs2(x1 - xdata1[ind,:]) - tmp
    end
    # add!(w, tmp, n)
    w .= exp.(w)

    nothing
end

# Epanechnikov Kernels
# ____________________
# Second order Epanechnikov ekernel
function ekernel2(x::Real, xdata::RealVector, h::Real, w::Vector, n::Int)
    ind = 1
    ind_end = 1+n
    @inbounds while ind < ind_end
        u = (x - xdata[ind]) / h
        w[ind] = ifelse(abs(u)>=1.0, 0.0, 1-u*u)
        ind += 1
    end
    multiply!(w, 0.75 / h)
    nothing
end
function ekernel2(x::RealVector,xdata::RealVector,h::RealVector,w::Vector,n::Int)
    ind = 1
    d = 1
    ind_end = 1+n
    u_all = (x' .- xdata) ./ h';
    @inbounds while ind < ind_end
        u = u_all[ind,:];
        w[ind] = ifelse(abs2(u)>=1.0, 0.0, 1-abs2(u))
        ind += 1
    end
    multiply!(w, 0.75 /prod(h))
    nothing
end

# Fourth order Epanechnikov ekernel
function ekernel4(x::Real, xdata::RealVector, h::Real, w::Vector, n::Int)
    ind = 1
    ind_end = 1+n
    @inbounds while ind < ind_end
        u = (x - xdata[ind]) / h
        w[ind] = ifelse(abs(u)>=1.0, 0.0, 7*(abs2(u)^2)-10*abs2(u)+3)
        ind += 1
    end
    multiply!(w, .46875 / h)
    nothing
end
function ekernel4(x::RealVector,xdata::RealVector,h::RealVector,w::Vector,n::Int)
    ind = 1
    d = 1
    ind_end = 1+n
    u_all = (x' .- xdata) ./ h';
    @inbounds while ind < ind_end
        u = u_all[ind,:];
        w[ind] = ifelse(abs2(u)>=1.0, 0.0, 7*(abs2(u)^2)-10*abs2(u)+3)
        ind += 1
    end
    multiply!(w, 0.46875 /prod(h))
    nothing
end

# Kernel Type
#____________________


mutable struct Kernel
    xdata::Union{Real,RealVector}
    y::RealVector #Realized value
    n::Int #Number of observation
    h::Union{Real,RealVector} #bandwidth
    kern::Function
    estimate::Function
    forecast::Function
    function Kernel(xdata::Union{Real,RealVector},y::RealVector)
        # Constructor for the kernel function
        n = size(xdata)[1];
        d = size(xdata)[2];
        h =  ones(d);
        kern_estimate = x->estimate(x,Kern)
        kern_forecast = x->forecast(x,Kern)
        return(new(xdata,y,n,h,ekernel4,kern_estimate,kern_forecast));
    end
end

function estimate(x::Union{Real,RealVector},Kern::Kernel)
    w = zeros(Kern.n);
    Kern.kern(x,Kern.xdata,Kern.h,w,Kern.n);
    w_diag = diagm(0=>w);
    β_kernel = inv(Kern.xdata'*w_diag * Kern.xdata) * (Kern.xdata' * w_diag * Kern.y);
    return(β_kernel);
end

function forecast(x::Union{Real,RealVector},Kern::Kernel)
    β_kernel = estimate(x,Kern);
    return(x' * β_kernel);
end

function forecast_fit(Kern::Kernel)
    yfit = zeros(Kern.n)
    for i = 1:Kern.n
        x_i = Kern.xdata[i,:];
        yfit[i] = Kern.forecast(x_i);
    end
    return(yfit);
end
