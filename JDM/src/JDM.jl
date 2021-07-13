module JDM

import CSV
using DataFrames
using DataFramesMeta
using Dates: Date, Time, unix2datetime
using Optim: BFGS, Fminbox, Options, optimize, minimizer
using ProgressMeter
using RollingFunctions: rollmean, rollstd
using Statistics: mean, std
using TimeZones

include("data.jl")

function jump_threshold(X::AbstractVector; k=2)
    μ = mean(X)
    σ = std(X)
    μ + k*σ
end

function log_likelihood(X, θ, σ)
    N = length(X)-1
    δ = 1 / (391 * 252)
    σ̃ = σ*√((1-exp(-2θ*δ))/(2θ))
    -N / 2 * log(2π) - N * log(σ̃) - 1/(2σ̃^2) * sum((X[t+1] - X[t] * exp(-θ*δ))^2 for t in 1:N)
end

function find_parameters(X::AbstractVector) 
    opt = coef -> -log_likelihood(X, coef...)
    lower = [0.0, 0.0]
    upper = [Inf, 10.0]
    init = [1.0, 1.0]
    solve = optimize(opt, lower, upper, init, Fminbox(BFGS()), Options(time_limit = 0.1), autodiff = :forward)
    minimizer(solve)
end

function backtest(data, starting_cash = 0.0; fee = 0.0005, initialization_period = 10, training_period = 30, trading_period = 5, num_pairs = 10)
    formation_period = initialization_period + training_period
    dates = unique(Date.(data.datetime))
    N = size(data, 2) - 1
    T = length(dates)
    cash = [starting_cash]
    @showprogress "Date: " for date in formation_period:T-trading_period-1
        training_dates = dates[date-formation_period+1:date]
        initialization_data = view(data, (Date.(data.datetime) .>= training_dates[1]) .& (Date.(data.datetime) .<= training_dates[initialization_period]), :)
        training_data = view(data, (Date.(data.datetime) .>= training_dates[initialization_period + 1]) .& (Date.(data.datetime) .<= training_dates[formation_period]), :)
        thresholds = zeros(N * (N-1) ÷ 2)
        pairs = Vector{Tuple{Int, Int}}(undef, N * (N-1) ÷ 2)
        Θ = similar(thresholds)
        Σ = similar(thresholds)
        i = 1
        @showprogress "Training loop: " 1 for j1 in 3:N+1, j2 in 2:j1-1
            pairs[i] = (j1, j2)
            initialization_spread = (view(initialization_data, :, j1) .- initialization_data[1, j1]) .- (view(initialization_data, :, j2) .- initialization_data[1, j2])
            opens = Time.(initialization_data.datetime) .== Time(9, 30)
            opens[1] =  0 # needed as we don't have a closing price before the first return
            overnight_variations = initialization_spread[opens] .- initialization_spread[vcat(opens[2:end], false)]
            thresholds[i] = jump_threshold(overnight_variations)
            training_spread = (view(training_data, :, j1) .- initialization_data[1, j1]) .- (view(training_data, :, j2) .- initialization_data[1, j2])
            opens = Time.(training_data.datetime) .== Time(9, 30)
            variations = training_spread[2:end] .- training_spread[1:end-1]
            filtered_variations = abs.(variations .* opens[2:end]) .> thresholds[i]
            filtered_spread = cumsum(vcat(training_spread[1], variations[.!filtered_variations]))
            if length(filtered_spread) < 1955
                Θ[i] = -Inf
                Σ[i] = Inf
            else
                demeaned_spread = filtered_spread[1955:end] .- rollmean(filtered_spread, 1955) # 1955 == 391*5
                θ, σ = find_parameters(demeaned_spread)
                Θ[i] = θ
                Σ[i] = σ
            end
            i+=1
        end
        asset_idx = partialsortperm(Θ, 1:num_pairs, rev=true)
        asset_pairs = get.(Ref(pairs), asset_idx, nothing)
        signals = zeros(trading_period*391, num_pairs)
        start_idx = findfirst(Date.(data.datetime) .== dates[date+1])
        end_idx = findlast(Date.(data.datetime) .== dates[date+trading_period])
        for (pair_idx, pair) in enumerate(asset_pairs)
            pair_data = view(data, (start_idx .- 1955):end_idx, [pair...])
            spread = (view(pair_data, :, 1) .- pair_data[1, 1]) .- (view(pair_data, :, 2) .- pair_data[1, 2])
            μ = 1/(1955 * Θ[asset_idx[pair_idx]]) .* (spread[1955:end] .- spread[1:end-1955+1] ) .+ rollmean(spread, 1955)
            σ = rollstd(spread, 1955)
            signal = zeros(length(spread) - 1955)
            for (i, (z, μ, σ)) in enumerate(zip(spread[1955:end], μ, σ))
                if i == 1
                    continue
                end
                if signals[i-1, pair_idx] == 0 && z > μ + 2σ
                    signals[i:end-1, pair_idx] .= -1
                elseif signals[i-1, pair_idx] == 0 && z < μ - 2σ
                    signals[i:end-1, pair_idx] .= 1
                elseif signals[i-1, pair_idx] == 1 && z > μ
                    signals[i:end-1, pair_idx] .= 0
                elseif signals[i-1, pair_idx] == -1 && z < μ
                    signals[i:end-1, pair_idx] .= 0
                end
            end
        end
        position = zeros((size(signals, 1), N))
        cash_change = 0
        investment = cash[end] / num_pairs
        for t in 2:size(signals, 1)
            for j in 1:size(signals, 2)
                if signals[t, j] != signals[t-1, j]
                    # signal changed, we invest
                    if signals[t, j] == 1
                        position[t:end, asset_pairs[j][1]-1] .+= (investment / exp(data[start_idx + t - 1, asset_pairs[j][1]])) * (1 - fee)
                        position[t:end, asset_pairs[j][2]-1] .-= (investment / exp(data[start_idx + t - 1, asset_pairs[j][2]])) * (1 - fee)
                    elseif signals[t, j] == -1
                        position[t:end, asset_pairs[j][1]-1] .-= (investment / exp(data[start_idx + t - 1, asset_pairs[j][1]])) * (1 - fee)
                        position[t:end, asset_pairs[j][2]-1] .+= (investment / exp(data[start_idx + t - 1, asset_pairs[j][2]])) * (1 - fee)
                    else
                        # closing out position
                        if signals[t-1, j] == 1
                            # Used to have long position
                            cash_change += (position[t, asset_pairs[j][1]-1] * exp(data[start_idx + t - 1, asset_pairs[j][1]])) * (1 - fee)
                            cash_change += (position[t, asset_pairs[j][2]-1] * exp(data[start_idx + t - 1, asset_pairs[j][2]])) * (1 + fee)
                        else
                            # Used to have short position
                            cash_change += (position[t, asset_pairs[j][1]-1] * exp(data[start_idx + t - 1, asset_pairs[j][1]])) * (1 + fee)
                            cash_change += (position[t, asset_pairs[j][2]-1] * exp(data[start_idx + t - 1, asset_pairs[j][2]])) * (1 - fee)
                        end
                        position[t:end, asset_pairs[j][1]-1] .= 0
                        position[t:end, asset_pairs[j][2]-1] .= 0
                    end
                end
            end
        end
        push!(cash, cash[end] + cash_change)
    end
    cash
end

end # module
