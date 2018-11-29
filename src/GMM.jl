




# Generate IV regression data
begin
      n = 3000;
      sig = [1 0.5 0.5;
             0 0.5 0 ;
             0 0  1];

      mu = [1,9,0];
      mn = MvNormal(mu,sig * sig')
      W = rand(mn,n)'
      x = W[:,1];z=W[:,2];ϵ=W[:,3];
      y = x * 3 + ϵ;
      β_OLS = inv(x'*x)*(x'*y);
      β_IV = inv(z'*x)*(z'*y);
      w = hcat(x,y,z);
end

function apply_row(func::Function, w::Array{Float64,2})
      δ = zeros(size(w,1));
      for (index, value) in enumerate(w[:,1])
           δ[index] = func(w[index,:]);
      end
      return(δ);
end

obj = θ->(begin
      η = (w[:,1] .* θ - w[:,2]) .* w[:,3];
      η'*η
      end)

using Optim
using Optim: converged, maximum, maximizer, minimizer, iterations #some extra functions
begin "Test Optim"
      result = optimize(obj,zeros(1),Newton(),autodiff = :forward)
      # @show converged(result) || error("Failed to converge in $(iterations(result)) iterations")
      @show xmin = result.minimizer
      @show result.minimum
end
