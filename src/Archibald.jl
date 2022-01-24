module Archibald
    using Distributions, StatsBase, LinearAlgebra
    using DataFrames, DataFramesMeta


    export MarkovChain
    export MetropolisHastings!, MetropolisHastings
    export tidybayes

    abstract type Markovian end
    """
    We create a MarkovChain object with a vector of parameters, a data vector, and 
    parametric type target_pdf which will be some sort of function. Using parametric 
    types is much more efficient than creating a field with abstract type Function 
    here.
    """
    struct MarkovChain{F<:Function} <: Markovian
        parameters::Vector
        data::Vector
        target_pdf::F
    end

    struct MarkovChainOverSample{F<:Function} <: Markovian
        single_sample_parameters::Vector
        S::Int64
        oversample_parameters::Matrix
        data::Vector
        target_pdf::F
    end
    include("samplers.jl")
    include("utils.jl")
end
