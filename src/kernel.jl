RealVector = Union{Array{Float64},Array{Real},Array{Int}}
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

function factorial2(x::Int)
    k = x;
    prod = 1;
    while (k > 0)
        prod *= k;
        k-=2;
    end
    return(prod);
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
function abs2(x::Array{Float64,2})
    return [x[i,:]'*x[i,:] for i=1:size(x,1)]
end
# # Second order gaussiankernel
# function gaussiankernel(x::Real, xdata::RealVector, h::Real, w::Vector, n::Int)
#     h1= 1.0/h
#     tmp = log(h) + log2π/2
#     for ind in 1:n
#         @inbounds w[ind]=-0.5*abs2((x - xdata[ind])*h1) - tmp
#     end
#     w .= exp.(w)
#
#     nothing
# end
#
# # MultiVariate prototype
# function gaussiankernel(x::RealVector, xdata::RealVector, h::RealVector, w::Vector, n::Int)
#     h1= 1.0/h
#     tmp = log(prod(h)) + log2π/2
#     x1 =x;
#     xdata1 = xdata;
#     for d = 1: size(h,1)
#         @inbounds x1[d] = x[d] / h[d]
#         @inbounds xdata1[:,d] = xdata[:,d] / h[d]
#         d+=1
#     end
#
#     for ind in 1:n
#         @inbounds w[ind]=-0.5*abs2(x1 - xdata1[ind,:]) - tmp
#     end
#     # add!(w, tmp, n)
#     w .= exp.(w)
#
#     nothing
# end





# Kern Function
#____________________
kern_epan_dict = Dict(2=> u->ifelse(abs2(u)>=1.0, 0.0, 0.75*(1-abs2(u))),
                      4=> u->ifelse(abs2(u)>=1.0, 0.0, 15/32*(7*(abs2(u)^2)-10*abs2(u)+3)),
                      6=> u->ifelse(abs2(u)>=1.0, 0.0, 105/256*(-33*(abs2(u)^3)+63*(abs2(u)^2)-35*abs2(u)+5)));
kern_biw_dict = Dict();
kern_triw_dict = Dict();
kern_gaussian_dict = Dict(2=> u-> (1 /sqrt(2*π)) * exp(-abs2(u)/2),
                    4=> u-> ifelse((3 - abs2(u))<=0, 0.0, 1 / (2*sqrt(2*π)))*(3 - abs2(u))*exp(-abs2(u)/2),
                    6=> u-> ifelse((15-10*abs2(u)+abs2(u)^2)<=0,0.0 ,1 / (8*sqrt(2*π)))*(15-10*abs2(u)+abs2(u)^2)*exp(-abs2(u)/2) );
kern_dict = Dict(:epan=>kern_epan_dict,:biw=>kern_biw_dict,
                :triw=>kern_triw_dict,:gaussian=>kern_gaussian_dict)
# Bandwidth
#____________________
κ_epan_dict = Dict(2=>1/5,4=>-1/21,6=>5/429);
κ_biw_dict  = Dict(2=>1/7,4=>-1/33,6=>1/143);
κ_triw_dict = Dict(2=>1/9,4=>-3/143,6=>1/221);
κ_gaussian_dict = Dict(2=>1.0,4=>-3.0,6=>15.0 );
κ_dict = Dict(:epan=>κ_epan_dict,:biw=>κ_biw_dict,:triw=>κ_triw_dict,:gaussian=>κ_gaussian_dict)

R_epan_dict = Dict(2=>3/5,4=>5/4,6=>1575/832);
R_biw_dict  = Dict(2=>5/7,4=>805/572,6=>29295/14144);
R_triw_dict = Dict(2=>350/429,4=>3780/2431,6=>301455/134368);
R_gaussian_dict = Dict(2=>1.0/(2 * sqrt(π)),4=>27 / (32 * sqrt(π)),6=>2265/(2048 * sqrt(π) ));
R_dict = Dict(:epan=>R_epan_dict,:biw=>R_biw_dict,:triw=>R_triw_dict,:gaussian=>R_gaussian_dict)

function bw_constant(q::Int,ν::Int,k_type::Symbol)
# Hansen, B. E. (2009). Lecture Notes on Nonparametrics.
# Retrieved from https://www.ssc.wisc.edu/~bhansen/718/NonParametrics1.pdf
    if !(ν in [2,4,6])    error("Kernel order has to be 2,4,6") end
    # if !(k_type in [:epan,:biw,:triw,:gaussian])    error("Kernel order has to be :epan,:biw,:triw,:gaussian")
    κ = κ_dict[k_type][ν];
    R = R_dict[k_type][ν];
    if q > 1
        Cₖ = ( (π^(q/2))*(2^(q+ν-1))*((factorial(ν))^2)*(R^q));
        Cₖ /= (ν*(κ^2)*(factorial2(2*ν-1)+(q-1)*(factorial2(ν-1)^2)));
        Cₖ = Cₖ^(1/(2*ν + q));
    elseif q == 1
         Cₖ = π^(1/2)*(factorial(ν)^3)*R;
         Cₖ /= (2*ν*factorial(2*ν)*κ^2);
         Cₖ = Cₖ^(1/(2*ν+1));
         Cₖ *= 2;
    else
        error();
    end
    return(Cₖ)
end

function bw_compute(σ::Union{Real,RealVector},n::Int, q::Int,ν::Int,k_type::Symbol)
    Cₖ = bw_constant(q,ν,k_type);
    bw = σ / Cₖ;
    bw *= n^(-1/(2*ν+q));
    return(bw);
end
# Kernel Type
#____________________


mutable struct Kernel
    order::Int
    type::Symbol
    kern::Function
    function Kernel(k_type::Symbol,ν::Int)
        # Constructor for the kernel function
        if !(ν in [2,4,6]) error("Kernel order has to be 2,4,6") end
        kern=kern_dict[k_type][ν];
        return(new(ν,k_type,kern));
    end
    # TODO: Can add self-defined kernel function
end

# Fourth order Epanechnikov ekernel
function compute_w(x::Real, xdata::RealVector, h::Real, w::Vector, n::Int, kern::Function)
    ind = 1
    ind_end = 1+n
    @inbounds while ind < ind_end
        u = (x - xdata[ind]) / h
        w[ind] = kern(u);
        ind += 1
    end
    multiply!(w, 1 / h)
    nothing
end
function compute_w(x::RealVector,xdata::RealVector,h::RealVector,w::Vector,n::Int,kern::Function)
    ind = 1
    d = 1
    ind_end = 1+n
    u_all = (x' .- xdata) ./ h';
    @inbounds while ind < ind_end
        u = u_all[ind,:];
        w[ind] =  kern(u);
        ind += 1
    end
    multiply!(w, 1 /prod(h))
    nothing
end

# Fourth order Epanechnikov ekernel
ekernel4 = (x,xdata,h,w,n)->compute_w(x,xdata,h,w,n,kern_dict[:epan][4])
ekernel2 = (x,xdata,h,w,n)->compute_w(x,xdata,h,w,n,kern_dict[:epan][2])
# ApproxFn
#____________________


mutable struct ApproxFn
    xdata::Union{Real,RealVector}
    y::RealVector #Realized value
    n::Int #Number of observation
    h::Union{Real,RealVector} #bandwidth
    q::Int #Number of regressors
    kern::Kernel #Kernel type
    function ApproxFn(xdata::Union{Real,RealVector},y::RealVector,k_type::Symbol,ν::Int)
        n = size(xdata,1);
        q = size(xdata,2);
        if length(y) != n
            error("The dimension of x,y must match");
        end
        if (q>1)
            σ = sqrt.(diag(cov(xdata)));
        elseif (q==1)
            σ = sqrt(cov(xdata));
        end
        h=bw_compute(σ,n,q,ν,k_type);
        if (q==1)
            h=h[1];
        end
        # self = new(x,y,n,"SVR",SVR(kernel="rbf",degree=4,gamma="scale"));
        # self = new(x,y,n,"KNN",KNeighborsRegressor());
        # fit!(self.model,x,y);
        kern=Kernel(k_type,ν);
        self=new(xdata,y,n,h,q,kern);
        return(self)
    end
    function (self::ApproxFn)(x::Union{Real,RealVector})
        if ((typeof(x)<:Real) & (self.q==1))
            return(forecast(x,self));
        elseif (typeof(x)<:Vector)& (self.q>1) &(size(x)[1]==self.q)
            return(forecast(x,self));
        elseif (typeof(x)<:Vector)&(self.q==1)
            forecast_y = zeros(size(x)[1]);
            for i = 1:size(x)[1]
                forecast_y[i] = forecast(x[i],self)[1];
            end
            return(forecast_y);
        elseif (self.q>1)&(size(x)[2]==self.q)
            forecast_y = zeros(size(x)[1]);
            for i = 1:size(x)[1]
                forecast_y[i] = forecast(x[i,:],self);
            end
            return(forecast_y);
        else
            error("Dimension of input incorrect")
        end
    end
    # function (self::ApproxFn)(x_in::Real)
    #     y_in = predict(self.model,hcat([x_in]));
    #     return(y_in)
    # end
    #
    # function (self::ApproxFn)(x_in::RealVector)
    #     y_in = predict(self.model,x_in);
    #     return(y_in)
    # end

end

function find_nn(neighbour_dist::Array{Float64,1},n::Int)
    neighbour_dist_sort = sort(neighbour_dist); #Sort from smallest to largest
    index=zeros(n);
    for i = 1:n
         for j = 1:length(neighbour_dist_sort)
             if (neighbour_dist_sort[i]==neighbour_dist[j])
                index[i]=j;
            end
         end
    end
    return(convert(Array{Int,1},index));
end

function estimatenn(x::Union{Real,RealVector},af::ApproxFn)
    # Use 10 nearest neighbour, I don't know, may be should do more
    dist2=y->abs2(x-y);
    n=10;
    neighbour_dist = apply_row(dist2,af.xdata);
    nn_index=find_nn(neighbour_dist,n);
    w=zeros(af.n);
    for i in nn_index
        w[i]=1/n;
    end
    w_diag = diagm(0=>w);
    β_kernel = inv(af.xdata'*w_diag * af.xdata) * (af.xdata' * w_diag * af.y);
    return(β_kernel);
end
# The ApproxFn
# The nonparametric function such that Y = ApproxFn(X)
function estimate(x::Union{Real,RealVector},af::ApproxFn)
    w = zeros(af.n);
    compute_w(x,af.xdata,af.h,w,af.n,af.kern.kern);
    if sum(w)==0
        return(estimatenn(x,af));
    else
        w_diag = diagm(0=>w);
        β_kernel = inv(af.xdata'*w_diag * af.xdata) * (af.xdata' * w_diag * af.y);
        return(β_kernel);
    end
end

function forecast(x::Union{Real,RealVector},af::ApproxFn)
    β_kernel = estimate(x,af);
    return(x' * β_kernel);
end

function UpdateData(self::ApproxFn,xdata,y)
    self.xdata = xdata;
    self.y     = y;
    fit!(self.model,self.x,self.y);
    return(self)
end

function UpdateVal(self::ApproxFn,y)
    self.y     = y;
    # fit!(self.model,self.x,self.y);
    return(self)
end





function forecast_fit(Kern::Kernel)
    yfit = zeros(Kern.n)
    for i = 1:Kern.n
        x_i = Kern.xdata[i,:];
        yfit[i] = Kern.forecast(x_i);
    end
    return(yfit);
end
