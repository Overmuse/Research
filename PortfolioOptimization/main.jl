using DataFrames, DotEnv, LibPQ, LinearAlgebra, Statistics
DotEnv.config()

conn = LibPQ.Connection("host=$(ENV["AIAIADB_HOST"]) user=postgres password=$(ENV["AIAIADB_PASSWORD"]) dbname=aiaiadb")
query = "SELECT * FROM adjusted_prices WHERE datetime >= '2016-06-04'"
res = DataFrame(execute(conn, query))
close(conn)
res = transform(groupby(res, :ticker), :close => (x -> vcat(missing, x[2:end] ./ x[1:end-1])) => :return)
returns = unstack(res[:, [:datetime, :ticker, :return]], :ticker, :return)
cov(Matrix(returns[2:end, 2:end]))
