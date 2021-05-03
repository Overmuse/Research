struct SyntheticAsset
    intercept::Float64
    betas::Vector{Float64}
    assets::Vector{Int}
    X::Matrix{Float64}
end

intercept(s::SyntheticAsset) = s.intercept
betas(s::SyntheticAsset) = s.betas
function price(s::SyntheticAsset, X = s.X)
    X[:, s.assets] * s.betas .+ s.intercept
end

function SyntheticAsset(X, y; n = nothing)
    assets = limit_assets(X, y; n = n)
    intercept, betas... = coef(lm(hcat(ones(length(y)), X[:, assets]), y))
    SyntheticAsset(intercept, betas, assets, X)
end

#' Performs multiple regression analyses to limit the number of assets considered for cointegration
function limit_assets(X, y; n=nothing)
    if isnothing(n)
        error("Need to specify the number of assets to limit to")
    end
    if n >= size(X, 2)
        return collect(1:size(X, 2))
    end

    assets = Int[]

    m = y
    for i in 1:n
        idx = next_asset(view(X, :, Not(assets)), m)
        idx = collect(1:size(X, 2))[Not(assets)][idx]
        push!(assets, idx)
        β = hedging_betas(view(X, :, assets), m)
        m = y - view(X, :, assets) * β
    end
    assets
end

function next_asset(X, y)
    max_r² = -Inf
    idx = 0
    for i in 1:size(X, 2)
        r² = rsquared(y, view(X, :, i) \ y .* view(X, :, i))
        if r² > max_r²
            max_r² = r²
            idx = i
        end
    end
    idx
end

function rsquared(y, x)
    cor(x, y)^2
end

function variance_ratio(y, τ)
    Δᵗy = y[(τ+1):end] .- y[1:(end-τ)]
    Δy = y[2:end] .- y[1:end-1]
    sum((Δᵗy .- mean(Δᵗy)).^2) / (τ * sum((Δy .- mean(Δy)).^2))
end
