using LinearAlgebra, Statistics, JuMP, Cbc, GLM, Polygon, ProgressMeter, DataFrames, CovarianceEstimation, StatsBase, DataFramesMeta, LibPQ

function factors(X, n)
    Σ = cov(AnalyticalNonlinearShrinkage(), X)
    σ = sqrt.(diag(Σ))
    C = cov2cor(Matrix(Σ), σ)
    v = reverse(eigvecs(C), dims=2)[:, 1:n]
    q = v ./ σ
    F = X * q 
end

function decompose(X, n)
    F = factors(X, n)
    β = F \ X
    U = X .- F * β
    U
end

function fit_ou(U)
    n = size(U, 1)
    map(eachcol(U)) do u
        X = cumsum(u)
        model = lm(hcat(ones(n-1), X[1:end-1]), X[2:end])
        a, b = coef(model)
        ν = std(residuals(model))
        r² = r2(model)
        OrnsteinUhlenbeck(a, b, ν, r²)
    end
end

struct OrnsteinUhlenbeck
    success::Bool
    κ::Float64
    m::Float64
    σ::Float64
    σ₌::Float64
    r²::Float64

    function OrnsteinUhlenbeck(a, b, ν, r²)
        try
            κ = -log(b) * 252
            m = a / (1 - b)
            σ = √((ν^2 * 2κ)/(1 - b^2))
            σ₌ = √(ν^2 / (1 - b^2))
            new(true, κ, m, σ, σ₌, r²)
        catch e
            new(false, 0, 0, 0, 0, 0)
        end
    end
end

function choose_assets(models::Vector{OrnsteinUhlenbeck}, n; r²_cutoff = 0.9)
    sortperm(models, by = model -> model_score(model, r²_cutoff), rev = true)[1:n]
end

function signal(model::OrnsteinUhlenbeck, u; r²_cutoff = 0.0)
    if !model.success || model.r² < r²_cutoff
        [nothing]
    else
        (cumsum(u) .- mean(u)) ./ (std(u) / √model.κ)
    end
end

function trade_signal(s, position)
    if isnothing(s)
        sign(position)
    elseif (position == 0 && s > 1.25) || (position < 0 && s > 0.5)
        -1
    elseif (position == 0 && s < -1.25) || (position > 0 && s < -0.5)
        1
    else
        0
    end
end

function optimize(s, positions, β, w, leverage; opt = Cbc.Optimizer(logLevel = 0))
    n = length(s)
    p = size(β, 1)
    m = Model(() -> opt)
    U = Set{Int}()
    for (i, (signal, pos)) in enumerate(zip(s, positions))
        if sign(signal) == sign(pos)
            push!(U, i)
        end
    end
    C = setdiff(Set(1:n), U)
    @variable(m, q[1:n]) # amount invested in asset
    @variable(m, b[1:n], Bin)
    # q⁺ + q⁻ == |q|
    @variable(m, 0 <= q⁺[1:n])
    @variable(m, 0 <= q⁻[1:n])
    @variable(m, f[1:p]) # factor exposure
    # f⁺ + f⁻ == |f|
    @variable(m, 0 <= f⁺[1:p])
    @variable(m, 0 <= f⁻[1:p])
    for u in U
        @constraint(m, q[u] == positions[u])
    end
    for c in C
        if s[c] > 0
            @constraint(m, q[c] >= 0)
        elseif s[c] < 0
            @constraint(m, q[c] <= 0)
         else
            @constraint(m, q[c] == 0)
        end
    end
    @constraint(m, f .== f⁺ .- f⁻)
    @constraint(m, f⁺ .- f⁻ .- sum(β[:, i] * q[i] for i in 1:n) .== 0)
    @constraint(m, q .== q⁺ .- q⁻)
    @constraint(m, q⁺ .<= b .* leverage)
    @constraint(m, q⁻ .<= (1 .- b) .* leverage)
    @constraint(m, sum(q[i] for i in C) == -sum(q[i] for i in U))
    @constraint(m, sum(q⁺[i] + q⁻[i] for i in C) + sum(q⁺[i] + q⁻[i] for i in U) == leverage)
    @objective(m, Min, sum(w[k] * (f⁺[k] + f⁻[k]) for k in 1:p))
    optimize!(m)
    try
        value.(q)
    catch e
        # Failed to solve, so no investments
        positions
    end
end

function download_data()
    conn = LibPQ.Connection("user=postgres password=password host=localhost port=5432 dbname=data")
    data = LibPQ.execute(conn, "SELECT * FROM adjusted_prices WHERE datetime > '2010-01-01';") |> DataFrame
    spread = @linq data |>
        select(:ticker, :datetime, :close) |>
        unstack(:ticker, :close)
    indices = map(x -> !any(ismissing, x), eachcol(spread))
    full = disallowmissing(spread[:, indices])
    Matrix(full[2:end, 2:end] ./ full[1:end-1, 2:end] .- 1)
end

function backtest(ret_matrix; training_length = size(ret_matrix, 1) ÷ 2, n_assets = 10, n_factors = 5, lookback = 60, r²_cutoff = 0.75)
    T, n = size(ret_matrix)
    positions = zeros(T, n)
    K = zeros(training_length - lookback, n)
    @showprogress for t in training_length+1:lookback:T-lookback
        # training loop
        Threads.@threads for t1 in t-training_length:t-lookback-1
            X = view(ret_matrix, t1:t1+lookback-1, :)
            F = factors(X, n_factors)
            β = F \ X 
            residuals = X .-  F * β
            models = fit_ou(residuals)
            K[mod1(t1, training_length-lookback), :] .= getfield.(models, :κ)
        end
        for t2 in t-1:t+lookback-2
            model_idx = partialsortperm(vec(mean(K, dims = 1)), 1:n_assets, rev = true)
            X = view(ret_matrix, t2-lookback+1:t2, :)
            F = factors(X, n_factors)
            β =  F \ X 
            residuals = X .-  F * β
            signals = Vector{Union{Float64, Nothing}}(undef, n)
            signals .= 0
            models = fit_ou(residuals[:, model_idx])
            signals[model_idx] .= map(last, signal.(models, eachcol(residuals[:, model_idx]), r²_cutoff = r²_cutoff))
            trade_signals = trade_signal.(signals, positions[t2, :])
            trades = optimize(trade_signals, positions[t2, :], β[2:end, :], fill(1/n_factors, n_factors), 1000.0)
            positions[t2+1:min(t2+5, end), :] .= trades'
        end
    end
    positions
end
