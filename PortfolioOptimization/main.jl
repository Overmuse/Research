using
    BlackLitterman,
    CovarianceEstimation,
    DataFrames,
    Dates,
    DotEnv,
    LibPQ,
    LinearAlgebra,
    Statistics

DotEnv.config()

include("download_data.jl")
include("market_caps.jl")

returns = download_data()
returns = select(select(returns, Not(:yearmonth)), Not(:IGOV))
excess_returns = select(returns .- returns.SHV, Not(:SHV))
Σ = Matrix(cov(AnalyticalNonlinearShrinkage(), Matrix{Float64}(excess_returns)))
global_weights = map(x -> x / sum(values(MARKET_CAPS)), MARKET_CAPS)
weights = [
    global_weights[:US_IG_DEBT],
    global_weights[:DEVELOPED_DEBT],
    global_weights[:EMERGING_DEBT],
    global_weights[:US_HY_DEBT],
    global_weights[:DEVELOPED_EQUITY],
    global_weights[:US_EQUITY],
    global_weights[:EMERGING_EQUITY],
]
μ = market_expected_returns(Σ, weights)

# Views for dividend yields
