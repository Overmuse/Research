module StatArb
include("hedging_portfolio.jl")
include("dimensionality_reduction.jl")
#include("robust_stat_arb.jl")
include("eigen.jl")

using Distributions, InvertedIndices, LinearAlgebra, Turing
using GLM: coef, lm

@model function arma_garch(y, ::Type{TV}=Vector{Float64}) where {TV}
    T = length(y) 

    ω ~ Truncated(Cauchy(0, 1), 0, 1)
    α ~ TruncatedNormal(0, 1, 0, 1)
    β ~ TruncatedNormal(0, 1, 0, 1)
    μ ~ TruncatedNormal(0, 0.1, -0.1, 0.1)
    ϕ ~ TruncatedNormal(0, 1, -1, 1)
    θ ~ TruncatedNormal(0, 1, -1, 1)
    ν ~ TruncatedNormal(1, 3, 0, Inf)

    σ = TV(undef, T)
    ϵ = TV(undef, T)
    ŷ = TV(undef, T)

    σ[1] ~ TruncatedNormal(0,0.1,0,0.5)
    ŷ[1] ~ Normal(0, 0.1)
    ϵ[1] = y[1] - ŷ[1]

    for t in 2:T
        σ[t] = sqrt(ω + α * ϵ[t-1]^2 + β * σ[t-1]^2)
        ŷ[t] = μ + ϕ * y[t-1] + θ * ϵ[t-1]
        ϵ[t] = y[t] - ŷ[t]
        y[t] ~ LocationScale(ŷ[t], σ[t], TDist(ν))
    end
end


function draw_path(ω, α, β, μ, ϕ, θ, ν, σ₀, ŷ₀, T)
    y = Vector{Float64}(undef, T)
    y[1] = ŷ₀
    z = LocationScale(0, 1, TDist(ν))
    σ = σ₀
    ϵ = σ * rand(z)
    for t in 2:T
        σ = sqrt(ω + α * ϵ^2 + β * σ^2)
        y[t] = μ + ϕ * y[t-1] + θ * ϵ
        ϵ = σ * rand(z)
    end
    y
end
end # module
