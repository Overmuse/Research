using LinearAlgebra, Statistics, JuMP, Cbc, GLM, Polygon, ProgressMeter, DataFrames

function factors(X, n)
    C = cor(X)
    v = reverse(eigvecs(C), dims=2)[:, 1:n]
    q = v ./ std(X, dims = 1)'
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
        0
    else
        (cumsum(u) .- model.m) ./ (model.σ₌)
    end
end

function trade_signal(s, position)
    if position == 0 && s > 1.25
        -1
    elseif position == 0 && s < -1.25
        1
    elseif position > 0 && s > -0.5
        -1
    elseif position < 0 && s < 0.5
        1
    else
        0
    end
end

function optimize(s, β, w, leverage; opt = Cbc.Optimizer(logLevel = 0))
    n = length(s)
    p = size(β, 1)
    m = Model(() -> opt)
    U = Set(i for (i, x) in enumerate(s) if iszero(x))
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
    for i in 1:n
        if s[i] > 0
            @constraint(m, q[i] >= 0)
        elseif s[i] < 0
            @constraint(m, q[i] <= 0)
        else
            @constraint(m, q[i] == 0)
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
        zeros(n)
    end
end

function download_data(tickers, start_date, end_date)
    data = mapreduce((a, b) -> outerjoin(a, b, on = :t, makeunique=true), tickers) do ticker
        @info ticker
        dicts = get_historical_range(get_credentials(), ticker, start_date, end_date, 1, "day", adjusted=true)
        df = reduce(vcat, DataFrame.(dicts))
        select(df, :t, :c)
    end
    data_matrix = Matrix(data[:, 2:end])
    data_matrix[2:end, :] ./ data_matrix[1:end-1, :] .- 1
end

function backtest(ret_matrix; training_length = size(ret_matrix, 1) ÷ 2, n_assets = 10, n_factors = 5, lookback = 60, r²_cutoff = 0.9)
    positions = zeros(size(ret_matrix))
    p1 = Progress(length(lookback+1:1+training_length), desc = "Training")
    # training loop
    Κ = mapreduce(hcat, lookback+1:1:training_length) do t
        next!(p1)
        X = view(ret_matrix, t-lookback:t, :)
        F = factors(X, n_factors)
        β = F \ X 
        residuals = X .- F * β
        models = fit_ou(residuals)
        getfield.(models, :κ)
    end
    model_idx = sortperm(vec(mean(Κ, dims = 2)), rev = true)[1:n_assets]
    p2 = Progress(length(training_length+1:1:size(ret_matrix, 1)), desc = "Testing")
    # test loop
    for t in training_length+1:1:size(ret_matrix, 1)
        X = view(ret_matrix, t-lookback:t-1, :)
        F = factors(X, n_factors)
        β = F \ X
        residuals = X .- F * β
        models = fit_ou(residuals[:, model_idx])
        signals = map(last, signal.(models, eachcol(residuals[:, model_idx]), r²_cutoff = r²_cutoff))
        trade_signals = trade_signal.(signals, positions[t, model_idx])
        trades = optimize(trade_signals, β, fill(1/n_factors, n_factors), 1000.0)
        positions[t:end, model_idx] .+= trades'
        next!(p2)
    end
    positions
end
