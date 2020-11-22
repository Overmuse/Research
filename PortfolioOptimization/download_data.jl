function download_data()
    conn = LibPQ.Connection("host=$(ENV["AIAIADB_HOST"]) user=postgres password=$(ENV["AIAIADB_PASSWORD"]) dbname=aiaiadb")
    query = "SELECT * FROM adjusted_prices WHERE datetime >= '2016-06-04'"
    res = dropmissing(DataFrame(execute(conn, query)))
    close(conn)
    sort!(res, [:ticker, :datetime])
    res = transform(groupby(res, :ticker), :close => (x -> vcat(missing, x[2:end] ./ x[1:end-1])) => :return)

    returns = filter(res) do row
        row.datetime >= Date(2016,7,1) && row.datetime <= Date(2020,10,31)
    end
    returns[!, :yearmonth] = yearmonth.(returns.datetime)
    returns = combine(groupby(returns, [:ticker, :yearmonth]), :return => (x -> prod(x) .- 1) => :return)
    returns = unstack(returns[:, [:yearmonth, :ticker, :return]], :ticker, :return)
    return returns
end
