function make_dataframe(aggregates)
    maybe_missing_col = ["op", "a"]
    mapreduce(vcat, aggregates) do agg
        for col in maybe_missing_col
            if !haskey(agg, col)
                agg[col] = nothing
            end
        end
        DataFrame(agg)
    end
end

function add_datetime(df)
    zdt = ZonedDateTime.(unix2datetime.(df.t ./ 1000), tz"UTC")
    df[!, :datetime] =astimezone.(zdt, tz"America/New_York") 
    df
end

function filter_to_market_hours(df)
    filter(df) do row
        Time(row.datetime) >= Time(9,30) && Time(row.datetime) <= Time(16)
    end
end

function clean_data(aggregates)
    aggregates |>
        make_dataframe |> 
        add_datetime |>
        filter_to_market_hours
end
