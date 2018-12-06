function convert_data(data)
    a_t  = [];
    a_t1 = [];
    s_t  = [];
    s_t1 = [];
    for t = 1:(nT-1)
        a_t  = vcat(a_t,data.action[:,t]);
        s_t  = vcat(s_t,data.state[:,t]);
        a_t1 = vcat(a_t1,data.action[:,t+1]);
        s_t1 = vcat(s_t1,data.action[:,t+1]);
    end
    return(action_t=a_t,action_t_1=a_t1,state_t=s_t,state_t_1=s_t1);
end
