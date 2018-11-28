
begin
      n = 3000;
      sig = rand(3,3)
      mu = rand(3)
      mn = MvNormal(mu,sig * sig')
      x_m = rand(mn,n)';
      x = x_m[:,[1,2]]
      β₀ = [1,3];
      y = x * β₀ + x_m[:,3]
      @show mn.Σ.mat;
      @show cov(x);
      β_OLS  = inv(x'*x) *(x'*y)
end

# Generate IV regression data
begin
      n = 3000;
      sig = [1 0.5 0.5;
             0 0.5 0 ;
             0 0   1];

      mu = [1,9,0];
      mn = MvNormal(mu,sig * sig')
      W = rand(mn,n)'
      x = W[:,1];z=W[:,2];ϵ=W[:,3];
      y = x * 3 + ϵ;
      β_OLS = inv(x'*x)*(x'*y);

      w = hcat(x,y,z);
end

function apply_row(func::Function, w::Array{Float64,2})
      δ = zeros(size(w,1));
      for (index, value) in enumerate(w[:,1])
           δ[index] = func(w[index,:]);
      end
      return(δ)
end

function momment(w,θ)
      η = (w[1] * θ  - w[2]) * w[3]
      return η;
end

obj = θ->(begin
      η = apply_row(x->moment(x,θ[1]),w);
      η' * η;
      end)

using Optim
using Optim: converged, maximum, maximizer, minimizer, iterations #some extra functions
begin "Test Optim"
      result = optimize(obj,zeros(1))
      @show converged(result) || error("Failed to converge in $(iterations(result)) iterations")
      @show xmin = result.minimizer
      @show result.minimum
end

begin
      rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
      result = optimize(rosenbrock, zeros(2), BFGS())
end
