function log_returns(df)
    temp = df
    temp[!, :date] = Date.(df.datetime)
    temp = groupby(temp, :date)
    combine(temp, :datetime => (x -> x[2:end]) => :datetime, :c => (x -> log.(x[2:end]) .- log.(x[1:end-1])) => :log_returns, keepkeys=false)
end
