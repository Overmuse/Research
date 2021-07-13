function read_data(file = "output.csv")
    data = CSV.read(file, DataFrame)
    data[!, :datetime] = astimezone.(ZonedDateTime.(unix2datetime.(data.t ./ 1000), tz"UTC"), tz"America/New_York")
    data[!, :log_prices] = log.(data.c)
    data = @linq data |>
        transform(date = Date.(:datetime), time = Time.(:datetime)) |>
        groupby(:date) |>
        where(:time .>= Time(9, 30), :time .<= Time(16, 00)) |>
        select(:ticker, :datetime, :log_prices) |>
        unstack(:ticker, :log_prices)

    for j in 1:size(data, 2)
        for t in 2:size(data, 1)
            if ismissing(data[t, j])
                data[t, j] = data[t-1, j]
            end
        end
        for t in size(data,1)-1:-1:1
            if ismissing(data[t, j])
                data[t, j] = data[t+1, j]
            end
        end
    end
    disallowmissing(data)
end
