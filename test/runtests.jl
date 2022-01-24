using Archibald
using Test
using Distributions
using DataFrames
using LinearAlgebra



# @testset "Over Sampling MCMC" begin
    
    function generate_data(;n = 10, 
                            T = 100,
                            β = 0.25,
                            family = Bernoulli,
                            τ = 1,
                            link = x -> cdf(Normal(), x))
        i_index = repeat(1:n, inner = T)
        t_index = repeat(1:T, n)
        μ_x = rand(Normal(0, 1), n)
        α_i = rand(Normal(0, τ), n)
        x = vcat(rand.(Normal.(μ_x, 1), T )...)
        μ_vec = α_i[i_index] .+ x * β  
    
        y = rand.(family.(link.(μ_vec)))

        return y, x, i_index, t_index, α_i 
    end


    function linear_target_dist_oversample(y, 
                                           x, 
                                           i_index, 
                                           single_parameters,
                                           over_parameters)
        β = single_parameters[1]
        τ = exp(single_parameters[2])
        α_matrix = over_parameters
        S = size(α_matrix, 2)
        ll = Vector{Float64}(undef, S)
        i_index = Int.(i_index)
        for s in 1:S
            resid_vec = y .- α_matrix[i_index, s] .- x*β
            ll[s] = sum(log.(pdf.(Normal.(0, 1), resid_vec)))
        end
        τ_prior = TruncatedNormal(0, 3, 0, Inf)
        τ_prior_dens = pdf(τ_prior, τ)
        param_probability = sum(ll) + log(τ_prior_dens)
        return param_probability
    end
    
    
    
    
    
    
    sim_n = 5
    sim_t = 100
    sim_τ = 0.5
    linear_data = generate_data(n = sim_n, T = sim_t, τ=sim_τ, link = x -> Normal(x, 1), family = identity)

    oversample_parameters = randn(sim_n, 3)
    chain_over = Archibald.MarkovChainOverSample(
        [0.1, 0.2],
        3,
        oversample_parameters,
        [linear_data[1], linear_data[2], linear_data[3]],
        (x_single, x_over, data) -> linear_target_dist_oversample(data[1],
                                                   data[2],
                                                   data[3],
                                                   x_single, x_over)
    )
    MetropolisHastings!(
        chain_over,
        (x) -> MvNormal(x, 0.1),
        (x) -> MvNormal(sim_n, x[2]),
        logs = true
        )
    
    mvn_draws = MetropolisHastings(
        1_000, 
        mvn_markov_state, 
        x -> MvNormal(x, 0.1), # jeez what an elegant design right? right?
        logs = true, 
        burn = 1_000)
    # now "for real" with lower proposal variance
    mvn_df = tidybayes(mvn_draws...)
    @test isa(mvn_df, DataFrame)
    @test size(mvn_df, 1) == 106000
    @test isa(mvn_draws[1][1], Vector)
    @test isa(mvn_draws[1][1][1], Float64)
    @test !isnan(mvn_draws[1][1][2])

# end


@testset "Basic Metropolis Hastings" begin
    function generate_data(; N = 100)
        sigma_b = log(rand(InverseGamma(2, 2)))
        sigma_e = log(rand(InverseGamma(1, 1)))
        β = [1, 2, 3, 4]

        X = randn(2, 4, N)
        W = randn(2, 1, N)
        b = randn(1, N).*sigma_b
        e = randn(2, N).*sigma_e


        Y = Matrix{Float64}(undef, 2, N)
        for i in 1:N
            Y[:, i] = X[:, :, i] * β + W[:, :, i]*b[:, i] + e[:, i] 
        end
        return Y, X, W, b, β, e, sigma_b, sigma_e
    end

    """
    Create target distribution for regression problem
    """
    function mvn_lik(y, X, W, parameters)
        N = size(X, 3)
        β = parameters[1:4]
        # constrain sigmas to be positive using exp(.)
        σ_b = exp(parameters[5])
        σ_e = exp(parameters[6])

        ll_i = Vector{Float64}(undef, N)
        for i in 1:N
            # Account for variance from Wb_i
            Σ_sqrt = σ_b^2 * W[:, :, i]*W[:, :, i]' + σ_e^2 .* Matrix(I,2,2)
            Σ = (Σ_sqrt + Σ_sqrt')./2 # make it Hermitian
            resid = y[:, i] - X[:, :, i]*β
            ll_i[i] = log(pdf(MvNormal([0, 0], Σ), resid))
        end

        # sum log lik contribution
        ll = sum(ll_i)
        # priors
        σ_b_prior = pdf(InverseGamma(2, 2), σ_b)
        σ_e_prior = pdf(InverseGamma(1, 1), σ_e)

        P = ll + log(sum(pdf.(Normal(0, 10), β))) + log(σ_b_prior) + log(σ_e_prior) 
        return P
    end


    test_Y, test_X, test_W, test_b, test_β, test_e, test_σ_b, test_σ_e = generate_data(N = 1_00)

    # wrapper function to match arguments with previous P()
    mvn_target_pdf(x, data) =  mvn_lik(data[1], data[2], data[3], x) 

    # Save true values for comparison later
    true_values = vcat(test_β, exp(test_σ_b), exp(test_σ_e), test_b[:])[:]
    true_params = DataFrame(
        [
            :true_value => true_values,
            :term => "parameter_" .* string.(1:size(true_values, 1)),
            :term_name => [
                "β_1",
                "β_2",
                "β_3",
                "β_4",
                "σ_b",
                "σ_e",
                "b_" .* string.(1:(size(true_values, 1) - 6))...
            ]
        ]
    )

    # Initialise chain
    mvn_markov_state = MarkovChain(
        fill(1.0, size(true_params, 1)), 
        [test_Y, test_X, test_W], 
        mvn_target_pdf
    )
    # burn in ad hoc for 20k
    mvn_draws = MetropolisHastings(
        1_000, 
        mvn_markov_state, 
        x -> MvNormal(x, 0.1), # jeez what an elegant design right? right?
        logs = true, 
        burn = 1_000)
    # now "for real" with lower proposal variance
    mvn_df = tidybayes(mvn_draws...)
    @test isa(mvn_df, DataFrame)
    @test size(mvn_df, 1) == 106000
    @test isa(mvn_draws[1][1], Vector)
    @test isa(mvn_draws[1][1][1], Float64)
    @test !isnan(mvn_draws[1][1][2])
end
