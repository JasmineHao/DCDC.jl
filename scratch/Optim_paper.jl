
function ff(x::Array)
    dist1 =  x' * [1.0,0.0] - mean(rand(Normal(3,2),100))
    dist2 =  x' * [0.0,1.0] - mean(rand(Normal(3,28),100))
    return(dist1^2 + dist2^2)
end

function ff2(x::Array)
    return((x - [1,1])' * (x - [1,1]))
end
optimize(ff2,zeros(2))

function ff3(x::Array)
    return((x'*[1]-1)^2)
end

optimize(ff3,zeros(1),BFGS())
optimize(ff3,zeros(1),SimulatedAnnealing())
optimize(ff3,[0.1],[0.5],[0.2],SAMIN()) #Bounded
optimize(ff3,zeros(1),ParticleSwarm())
