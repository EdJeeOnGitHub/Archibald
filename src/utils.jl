"""
Clean up and extract MCMC_draws and MCMC_diagnostics from MetropolisHastings.
"""
function tidybayes(MCMC_draws, MCMC_diagnostics)
    param_df = DataFrame(MCMC_draws, :auto)
    K_terms = size(param_df, 1)
    param_df.term = "parameter_" .* string.(1:K_terms)
    param_df = DataFramesMeta.stack(param_df, Not(:term))
    transform!(param_df, :variable => ByRow(x -> parse(Int, split(x, "x")[2])) => :draw)
    select!(param_df, Not(:variable))

    draw_df = DataFrame(
        [
            :draw => 1:length(MCMC_draws),
            :accept => MCMC_diagnostics[:, 1]
        ])
        param_df
    all_draw_df = innerjoin(draw_df, param_df, on = :draw) 
    return all_draw_df
end
