function _moment(θ)
    (β,σ) = θ;
    util = Utility(σ);
    dutil = (s) -> Tracker.gradient(util,s)[1].data;
    dudc  = dutil.(a_t);
    dudc1 = dutil.(a_t1);
    dtrans=ddc.dtrans;
    dtrans_s = (a,s) -> dtrans(a,s)[2].data;

    return(dudc - β .* dtrans_s.(a_t1,s_t1) .* dudc1)
end




mutable struct Estimation
    data
    Vhat::ApproxFn
    Transition::Transition
    exWeight::Array{Float64,2}
    ltp1::Array{Int,1}
    invl::Array{Int,1}
    #  -ltp1[j] = location in sivec of observation for same firm at
    #  time t+1, it is < 0 if there is no such observation
    #  -invl[j] = location in sivec of observation for same firm at
    #   time t-1, it is < 0 if there is no such observation
    function Estimation(data)
        a=data.action,s=data.state;
        nT=size(a,2);nM=size(a,1);
        a_t = []; a_t1 = []; s_t = []; s_t1=[];
        a_t = [];
        a_t1 = [];
        s_t = [];
        for t = 1:(nT-1)
            a_t  = vcat(a_t,data.action[:,t]);
            s_t  = vcat(s_t,data.state[:,t]);
            a_t1 = vcat(a_t1,data.action[:,t+1]);
            s_t1 = vcat(s_t1,data.state[:,t+1])
        end
        # First estimate transition density


    end
        # for t = 1:(nT-1)
        #     global a_t  = vcat(a_t,data.action[:,t]);
        #     global s_t  = vcat(s_t,data.state[:,t]);
        #     global a_t1 = vcat(a_t1,data.action[:,t+1])
        # end

    end

end
