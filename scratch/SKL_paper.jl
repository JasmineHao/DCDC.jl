using RDatasets: dataset

iris = dataset("datasets", "iris")

# ScikitLearn.jl expects arrays, but DataFrames can also be used - see
# the corresponding section of the manual
X = convert(Array, iris[[:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]])
y = convert(Array, iris[:Species])


using ScikitLearn
using Distributions
@sk_import linear_model: LogisticRegression
# This model requires scikit-learn. See
# http://scikitlearnjl.readthedocs.io/en/latest/models/#installation
@sk_import linear_model: LogisticRegression
x = randn(300,4)
srand = (134)
ỹ = x * [3,-4,5,-1] + rand(Logistic(),300) ;
y =  ifelse.(ỹ .> 0, 1.0, 0.0)

model = LogisticRegression(fit_intercept=true)
fit!(model, x, y)
accuracy = sum(predict(model, x) .== y) / length(y)
println("accuracy: $accuracy")



using ScikitLearn.GridSearch: GridSearchCV
gridsearch = GridSearchCV(LogisticRegression(), Dict(:C => 0.1:0.1:2.0));
fit!(gridsearch, x, y)
println("Best parameters: $(gridsearch.best_params_)")

# Kernel Ridge
@sk_import kernel_ridge: KernelRidge
n = 300
x = randn(n,1);
y = x* [3] + 1 * rand(Logistic(),n)
model_krr = KernelRidge()
fit!(model_krr,x,y)
ŷ = predict(model_krr,x)
using Plots
scatter(x,y,line=(5,:dot,:green))
scatter!(x,ŷ)
