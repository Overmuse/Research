module DoubleOU

using BusinessDays: advancebdays, initcache, listbdays, isbday, USNYSE
using CSV
using DataFrames
using Dates
using Impute
using IterTools: product
using Optim: BFGS, Fminbox, optimize, minimizer
using Polygon: get_credentials, get_open_close
using ProgressMeter: @showprogress
using ResearchTools: add_datetime, filter_to_market_hours
using Roots: find_zero
using ShiftedArrays: ShiftedArray
using Statistics: quantile, var
using StatsBase: ordinalrank

export main

include("longterm.jl")
include("intraday.jl")

function spreads(data)
    m, n = size(data)
    out = zeros(m, n*(n-1)÷2)
    x = 1
    for i in 2:size(data, 2), j in 1:i-1
        out[:, x] = log.(data[:, j] ./ data[1, j]) .- log.(data[:, i] ./ data[1, i])
        x += 1
    end
    out
end 

function load_data(file)
    data = CSV.read(file, DataFrame)
    df = filter_to_market_hours(add_datetime(sort(unstack(select(data, :t, :ticker, :c), :ticker, :c), :t)))
	df.date = Date.(df.datetime)
    Impute.interp!(df, limit = 10)
	data = combine(vcat, filter(x -> size(x, 1) == 79, groupby(df, :date)))[:, 3:end]
end

function index_to_pair(idx)
    j = (1+isqrt(8idx-7))÷2+1
    i = idx - ((j-2)*(j-1))÷2
    (i, j)
end

struct DaySummary
    cash_change::Float64
    num_trades::Int
    winning_trades::Int
    fully_reverted_trades::Int
end

function main(data; n=50)
    dates = Date.(data.datetime) |> unique
    out = Vector{DaySummary}(undef, length(dates)-100)
    @showprogress for i in 100:length(dates)-1
        longterm_dates = dates[i-100+1:i+1]
        filtered_data = data[(Date.(data.datetime) .>= longterm_dates[1]) .& (Date.(data.datetime) .<= longterm_dates[end]), 1:end-1]
        filtered_data = disallowmissing(Impute.filter(filtered_data, dims = :cols))
        longterm_data = filtered_data[reduce(vcat, [[79*(j-1)+1, 79j] for j in 1:100]), :]
        longterm_spreads = spreads(longterm_data)
        L_params = calibrate_L.(eachcol(longterm_spreads))
        score_L = map(L_params) do (θ, _, σ)
            L_score(θ, σ)
        end
        shortterm_dates = dates[i-30+1:i]
        shortterm_data = filtered_data[end-31*79+1:end-79, :] 
        shortterm_spreads = spreads(shortterm_data)
        deltas = map(L_params) do (_, δ, _)
            δ
        end
        params = calibrate.(eachcol(shortterm_spreads), deltas)
        score_S = map(zip(params, deltas)) do ((θ, σ), δ₁)
            score(θ, σ, δ₁)
        end
        ranking = ordinalrank(score_L) .+ ordinalrank(score_S, rev = true)
        best_spreads = partialsortperm(ranking, 1:n)
        num_trades = 0
        winning_trades = 0
        fully_reverted_trades = 0
        cash_change = 0.0
        for idx in best_spreads
            pair = index_to_pair(idx)
            ϵ = quantile(abs.(longterm_spreads[2:end, idx] .- longterm_spreads[1:end-1, idx]), 0.95)
            trade_data = vcat(shortterm_data[:, [pair...]], filtered_data[end-78:end, [pair...]])
			trade_spread = vec(spreads(trade_data))
            mean_L = (trade_spread[2370] + trade_spread[2371]) / 2
            pos = 0
            shares = (0.0, 0.0)
            for t in 2371:2439 # 2349 because we close 10 minutes before end of day
                if (pos == 0) && (trade_spread[t] > mean_L + ϵ)
                    pos = -1
                    shares = (-1/trade_data[t, 1], 1/trade_data[t, 2])
                    num_trades += 1
                    cash_change -= 0.001
                elseif (pos == 0) && (trade_spread[t] < mean_L - ϵ)
                    pos = 1
                    shares = (1/trade_data[t, 1], -1/trade_data[t, 2])
                    cash_change -= 0.001
                    num_trades += 1
                elseif ((pos == -1) && (trade_spread[t] < mean_L)) || ((pos == 1) && (trade_spread[t] > mean_L))
                    pos = 0
                    cash_change += shares[1] * trade_data[t, 1] + shares[2] * trade_data[t, 2]
                    fully_reverted_trades += 1
                    winning_trades += 1
                    shares = (0.0, 0.0)
                end
            end
            cash_change_temp = shares[1] * trade_data[end, 1] + shares[2] * trade_data[end, 2]
            if cash_change_temp > 0.001
                winning_trades += 1
            end
            cash_change += cash_change_temp
        end
        out[i-100+1] = DaySummary(cash_change, num_trades, winning_trades, fully_reverted_trades)
    end
    out
end

end # module
