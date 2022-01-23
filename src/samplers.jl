
"""
Perform a MetropolisHastings step on a given Markovian state m, using a user 
supplied proposal function q. Typically q will use x_t as the proposal mean and 
generate x' using some pdf.
N.B. We never constrain variables to be positive/on some interval so this needs 
to be done using transforms in the m.target_pdf function. e.g. exp(σ) instead of
σ.
"""
function MetropolisHastings!(m::MarkovChain, q; logs = true)
    # rename everything from `parameters` to x and x' to match notes/maths
    x = m.parameters

    x_prime = rand(q(x), 1)[:, 1] # draw proposal 
    # evaluate g(x | x'). N.B. x' in first argument but conditioning 
    # on x' still
    g_x = pdf(q(x_prime), x) 
    g_x_prime = pdf(q(x), x_prime)
    

    # Calculate likelihood/probability whatever you want to call it at target 
    # and proposal params.
    P_x_prime = m.target_pdf(x_prime, m.data)
    P_x = m.target_pdf(x, m.data)
    # In exp/log space to prevent over/underflow
    # IF LOGS = TRUE THEN m.target_pdf BETTER BE IN LOGS EDWARD
    if logs == true
        A = exp(P_x_prime - P_x + log(g_x) - log(g_x_prime))
    else 
        A = (P_x_prime * g_x) / (P_x * g_x_prime)
    end
    # If accept, update params using x'. Otherwise, keep params == x.
    if rand() < A
        m.parameters[:] = x_prime
    end
    return m
end


"""
Call MetropolisHastings! N times with `burn` burn in draws on a given Markov 
state with proposal function q.
"""
function MetropolisHastings(N::Int, 
                            m::MarkovChain, 
                            q;
                            logs = true, 
                            burn = Int(floor(N/5)))
    MCMC_draws = Vector{Vector}(undef, N)
    MCMC_diagnostics = Matrix{Float64}(undef, N, 1)
    for i in 1:burn
        MetropolisHastings!(m, q, logs = logs)
    end
    old_params = m.parameters
    for i in 1:N
        new_params = MetropolisHastings!(m, q, logs = logs).parameters[:]
        MCMC_draws[i] = new_params
        MCMC_diagnostics[i, 1] = !isequal(old_params, new_params) 
        old_params = new_params
    end
    return MCMC_draws, MCMC_diagnostics
end


x = rand(10)[:, 1]

x
popat!.(Ref(x), idx)
idx = [2, 4, 5]
getindex(x, idx)

x[Not(idx)]
x[1:end .== idx]
x[!in.(idx)]kh

idx



"""
Perform a MetropolisHastings step on a given Markovian state m, using a user 
supplied proposal function q. Typically q will use x_t as the proposal mean and 
generate x' using some pdf.
N.B. We never constrain variables to be positive/on some interval so this needs 
to be done using transforms in the m.target_pdf function. e.g. exp(σ) instead of
σ.
"""
function MetropolisHastings!(m::MarkovChainOverSample, 
                             q, 
                             logs = true)
    K = size(m.parameters, 1)
    S = m.S
    oversample_params_idx = m.oversample_params_idx
    single_sample_params_idx = deleteat!(collect(1:K), oversample_params_idx)
    # rename everything from `parameters` to x and x' to match notes/maths
    x = m.parameters
    # No need to recompute this each time

    x_prime = rand(q(x), 1)[:, 1] # draw proposal 
    x_prime_s = similar(m.parameters)
    P_x_vec = m.target_pdf.(x, Ref(m.data))
    P_x_prime_vec = Vector{Float64}(undef, S)

    # N.B. g_x doesn't use over sampled param cond density.
    g_x = pdf(q(x_prime), x) 
    g_x_prime = pdf(q(x), x_prime)
    for s in 1:S
        params_s = rand(q(x), 1)[:, 1]
        x_prime_s[single_sample_params_idx] = x_prime[single_sample_params_idx]
        x_prime_s[oversample_params_idx] = params_s[oversample_params_idx] 
        # Calculate likelihood/probability whatever you want to call it at target 
        # and proposal params.
        P_x_prime_vec[s] = m.target_pdf(x_prime_s, m.data)
    end 

    # In exp/log space to prevent over/underflow
    # IF LOGS = TRUE THEN m.target_pdf BETTER BE IN LOGS EDWARD
    if logs == true
        A_s = exp(sum(P_x_prime_s) - sum(P_x) + log(g_x_s) - log(g_x_prime_s))
    else 
        A_s = (P_x_prime_s * g_x_s) / (P_x * g_x_prime_s)
    end
    if rand() < A_s
        m.parameters[]
end
    # If accept, update params using x'. Otherwise, keep params == x.

    accept = rand(S) .< A_s
    for 
    m.parameters[]


    if rand(S) .< A_s
        m.parameters[:] = x_prime
    end
    return m
end
