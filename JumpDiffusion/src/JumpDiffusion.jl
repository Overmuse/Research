module JumpDiffusion

using Distributions: Normal, quantile
using ProgressMeter: @showprogress
using SpecialFunctions: gamma

include("data.jl")
include("extension.jl")

function realized_variance(Y)
    sum(y^2 for y in Y)
end

function μ(x)
    2^(x/2) * gamma((x + 1)/2) / gamma(1/2)
end

function bipower_variation(Y)
    M = length(Y)
    μ(1)^(-2) * sum(abs(Y[j]) * abs(Y[j-1]) for j in 2:M)
end

function tripower_quarticity(Y)
    M = length(Y)
    M * μ(4/3)^(-3) * M / (M-2) * sum(abs(Y[j])^(4/3)*abs(Y[j-1])^(4/3)*abs(Y[j-2])^(4/3) for j in 3:M)
end

function zscore(Y)
    rv = realized_variance(Y)
    bpv = bipower_variation(Y)
    tp = tripower_quarticity(Y)
    M = length(Y)

    numerator = (rv - bpv) / rv
    denominator = √(((π/2)^2 + π - 5) / M * max(1, tp / bpv^2))
    numerator / denominator
end

function jump_term(Y)
    realized_variance(Y) - bipower_variation(Y)
end

function jump_index(Y)
    argmax(abs.(Y))
end

function jump_size(Y)
    idx = jump_index(Y)
    sign(Y[idx]) * jump_term(Y)
end

function jump_detected(Y, α = 0.001)
    Z = zscore(Y)
    abs(Z) > quantile(Normal(), 1 - α)
end

function choose_stocks(data, n=10)
    jump_stocks = jump_detected.(eachcol(data))
    eod_jump_stocks = (jump_index.(eachcol(data)) .== size(data, 1))
    largest_z = sortperm(zscore.(eachcol(data)), rev=true)
    chosen = Int[]
    for z in largest_z
        if length(chosen) == n
            return chosen
        end
        if eod_jump_stocks[z] && jump_stocks[z]
            push!(chosen, z)
        end
    end
    chosen
end

function backtest(data, starting_cash=1000.0; fee = 0.0005)
    dates = Date.(data.datetime) |> unique
    T = length(dates)
    cash = zeros(T)
    cash[1] = starting_cash
    @showprogress for t in 2:T
        train_data = @linq data |>
            where(DateTime.(:datetime) .>= dates[t-1] + Time(9, 30), DateTime.(:datetime) .<= dates[t] + Time(9, 30))
        for t2 in 2:size(train_data, 1), j in 1:size(train_data, 2)
            if ismissing(train_data[t2, j])
                train_data[t2, j] = train_data[t2-1, j]
            end
        end
        log_prices = log.(Matrix(train_data[:, 2:end])) # skip datetime
        log_returns = log_prices[2:end, :] .- log_prices[1:end-1, :]
        indices = map(x -> !any(ismissing, x), eachcol(log_returns))
        stocks_indices = findall(indices)[choose_stocks(disallowmissing(log_returns[:, indices]))]
        zs = abs.(zscore.(eachcol(log_returns[:, stocks_indices])))
        n = length(stocks_indices)
        jump_sign = sign.(log_returns[end, stocks_indices])
        test_data = @linq data |>
            where(DateTime.(:datetime) .>= dates[t] + Time(9, 30), DateTime.(:datetime) .<= dates[t] + Time(11, 30))
        testset = test_data[:, 1 .+ stocks_indices] # add one to account for date column
        # fill-forward prices
        for t2 in 2:size(testset, 1), j in 1:n
            if ismissing(testset[t2, j])
                testset[t2, j] = testset[t2-1, j]
            end
        end
        for t2 in size(testset, 1)-1:-1:1, j in 1:n
            if ismissing(testset[t2, j])
                testset[t2, j] = testset[t2+1, j]
            end
        end
        cash[t] = cash[t-1]
        for j in 1:n
            investment = zs[j] / sum(zs) * cash[t-1]
            #investment = cash[t-1] / n
            if jump_sign[j] < 0
                cash[t] += investment / testset[1, j] * testset[end, j] - investment - investment*2*fee
            else
                cash[t] += investment  - testset[end, j] * investment / testset[1, j] - investment*2*fee 
            end
        end
    end
    cash
end

end # module
