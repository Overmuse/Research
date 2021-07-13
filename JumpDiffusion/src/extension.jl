function build_testset(data; lookback=60, forward_eval=60)
    data.date = Date.(data.datetime)
    positive_jump_returns = []
    negative_jump_returns = []
    grouped = groupby(data, :date)
    @showprogress for group in grouped
        log_prices = log.(group[:, 2:end-1])
        log_returns = log_prices[2:end, :] .- log_prices[1:end-1, :]
        T = size(log_returns, 1)
        for t in lookback:T-forward_eval
            idx = choose_stocks(view(log_returns, t-lookback+1:t, :))
            for i in idx
                ret = log_returns[t, i]
                if ret > 0
                    push!(positive_jump_returns, (ret, log_returns[t:t+forward_eval, i]))
                else
                    push!(negative_jump_returns, (ret, log_returns[t+1:t+forward_eval, i]))
                end
            end
        end
    end
    (positive_jump_returns, negative_jump_returns)
end

function backtest_extension(data, starting_cash=1000.0; fee = 0.0005, lookback=60, forward_eval=60)
    @showprogress "Forward-filling data" for t in 2:size(data, 1), j in 2:size(data, 2)
        if ismissing(data[t, j])
            data[t, j] = data[t-1, j]
        end
    end
    @showprogress "Back-filling data" for t in size(data, 1)-1:-1:1, j in 2:size(data, 2)
        if ismissing(data[t, j])
            data[t, j] = data[t+1, j]
        end
    end
    data.date = Date.(data.datetime)
    grouped = groupby(data, :date)
    cash = [(data.datetime[1], starting_cash)]
    @showprogress for group in grouped
        log_prices = log.(Matrix(group[:, 2:end-1], ))
        log_returns = log_prices[2:end, :] .- log_prices[1:end-1, :]
        T = size(log_returns, 1)
        for t in lookback+1:T-forward_eval
            stocks_indices = choose_stocks(view(log_returns, t-lookback:t-1, :))
            positive_large_jump = log_returns[t-1, stocks_indices] .>= 0.01
            stocks_indices = stocks_indices[positive_large_jump]
            zs = abs.(zscore.(eachcol(view(log_returns, t-lookback:t-1, stocks_indices))))
            n = length(stocks_indices)
            cash_change = 0.0
            for j in 1:n
                investment = zs[j] / sum(zs) * cash[end][2]
                cash_change += investment - group[t+forward_eval, stocks_indices[j]+1] * investment / group[t, stocks_indices[j]+1] - investment*2*fee 
            end
            push!(cash, (group.datetime[t], cash[end][2] + cash_change))
        end
    end
    cash
end
