import CSV
using DataFrames
using Dates: Date, Time, unix2datetime
using DataFramesMeta
using TimeZones

function read_data(file = "output.csv")
    data = CSV.read(file, DataFrame)
    data[!, :datetime] = astimezone.(ZonedDateTime.(unix2datetime.(data.t ./ 1000), tz"UTC"), tz"America/New_York")
    data = @linq data |>
        select(:ticker, :datetime, :c) |>
        unstack(:ticker, :c) |>
        transform(date = Date.(:datetime), time = Time.(:datetime)) |>
        groupby(:date) |>
        where(:time .>= Time(9, 30), :time .<= Time(16, 00))
    data[:, 1:end-2]
end
