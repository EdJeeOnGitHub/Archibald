
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
                             f; 
                             logs = true)

    S = m.S
    # rename everything from `parameters` to x and x' to match notes/maths
    x_single = m.single_sample_parameters
    x_over = m.oversample_parameters
    # No need to recompute this each time

    x_prime_single = rand(q(x_single), 1)[:, 1] # draw proposal 
    x_prime_over = rand(f(x_prime_single), S)

    # N.B. g_x doesn't use over sampled param cond density.
    g_x = pdf(q(x_prime_single), x_single) 
    g_x_prime = pdf(q(x_single), x_prime_single)


    P_x_prime = m.target_pdf(x_prime_single, x_prime_over, m.data)
    P_x = m.target_pdf(x_single, x_over, m.data)


    # In exp/log space to prevent over/underflow
    # IF LOGS = TRUE THEN m.target_pdf BETTER BE IN LOGS EDWARD
    if logs == true
        A = exp(P_x_prime - P_x + log(g_x) - log(g_x_prime))
    else 
        A = (P_x_prime * g_x) / (P_x * g_x_prime)
    end

    if rand() < A
        m.single_sample_parameters[:] = x_prime_single
        m.oversample_parameters[:, :] = x_prime_over
    # If accept, update params using x'. Otherwise, keep params == x.
    end

    return m
end

function MetropolisHastings(N::Int, 
                            m::MarkovChainOverSample, 
                            q,
                            f;
                            logs = true, 
                            burn = Int(floor(N/5)))
    MCMC_draws = Vector{Vector}(undef, N)
    MCMC_diagnostics = Matrix{Float64}(undef, N, 1)
    for i in 1:burn
        MetropolisHastings!(m, q, f, logs = logs)
    end
    old_params = m.single_sample_parameters
    for i in 1:N
        new_params = MetropolisHastings!(m, q, f, logs = logs).single_sample_parameters[:]
        MCMC_draws[i] = new_params
        MCMC_diagnostics[i, 1] = !isequal(old_params, new_params) 
        old_params = new_params
    end
    return MCMC_draws, MCMC_diagnostics
end