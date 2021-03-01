### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 91e6fd18-6baf-11eb-32c6-ffff6529c2fb
using Pkg; Pkg.add("Plots")

# ╔═╡ d9d2b51e-6bae-11eb-231f-d9bf964b8c64
using ARCHModels, Polygon, Plots

# ╔═╡ f73b809a-6bae-11eb-1efc-ff48dccf5152
prices = Polygon.get_historical_range(get_credentials(), "AAPL", "2020-01-01", "2021-01-01")

# ╔═╡ 55260142-6baf-11eb-03c8-6190884721be
close_prices = [d["c"] for d in prices]

# ╔═╡ 7267b4f0-6baf-11eb-273c-3f7463ba4ad8
returns = log.(close_prices[2:end] ./ close_prices[1:end-1])

# ╔═╡ 89f49f8e-6baf-11eb-0212-edf9033171bb
plot(returns)

# ╔═╡ 5c35660e-6bb0-11eb-37bd-0bcbd7435632
function plot_model(returns, model)
	p1 = plot(returns, label = "Returns")
	plot!(p1, model, label = "Model")
	resid = returns .- model
	p2 = plot(resid, label = "Residual")
	plot(p1, p2, layout = (2, 1))
end

# ╔═╡ d0344ffa-6baf-11eb-10c8-e3186bd98ef4
md"""
## Model 1: Mean return
"""

# ╔═╡ b6427ed0-6baf-11eb-0f06-e39016c4a6c1
mean_return = mean(returns);

# ╔═╡ e7f9f796-6baf-11eb-343e-df27d7df80ca
plot_model(returns, fill(mean_return, length(returns)))

# ╔═╡ 808897e2-6bb0-11eb-2865-a533dcf49b12
md"""
## Model 2: SMA
"""

# ╔═╡ 8e6a1872-6bb0-11eb-3825-098258ba2745
sma = vcat(fill(missing, 12), [mean(returns[t:t+12]) for t in 1:(length(returns)-12)]);

# ╔═╡ b5e0b0a0-6bb0-11eb-23e1-b1dd987b548f
plot_model(returns, sma)

# ╔═╡ d7c4a62c-6bb0-11eb-39b1-8541451d8840
md"""
## Model 3: EMA
"""

# ╔═╡ ed34ff98-6bb0-11eb-14e4-71d769deba60
ema = map(1:length(returns)) do i
	if i == 1
		missing
	elseif i == 2
		returns[i-1]
	else
		0.1 * returns[i-2] * 0.9 * returns[i-1]
	end
end

# ╔═╡ 1308914e-6bb1-11eb-069a-ab093d004789
plot_model(returns, ema)

# ╔═╡ 7625367c-6bb1-11eb-1db9-49460415eb3e
md"""
## Model 4: ARCH
"""

# ╔═╡ 7e7e2fce-6bb1-11eb-3054-d329793998dd
begin 
	model = fit(GARCH{1, 1}, returns, meanspec = ARMA{1, 1})
	data = means(model)
end

# ╔═╡ e20e3002-6bbb-11eb-201e-7565164f9f11
plot_model(returns, data)

# ╔═╡ Cell order:
# ╠═91e6fd18-6baf-11eb-32c6-ffff6529c2fb
# ╠═d9d2b51e-6bae-11eb-231f-d9bf964b8c64
# ╠═f73b809a-6bae-11eb-1efc-ff48dccf5152
# ╠═55260142-6baf-11eb-03c8-6190884721be
# ╠═7267b4f0-6baf-11eb-273c-3f7463ba4ad8
# ╠═89f49f8e-6baf-11eb-0212-edf9033171bb
# ╠═5c35660e-6bb0-11eb-37bd-0bcbd7435632
# ╟─d0344ffa-6baf-11eb-10c8-e3186bd98ef4
# ╟─b6427ed0-6baf-11eb-0f06-e39016c4a6c1
# ╟─e7f9f796-6baf-11eb-343e-df27d7df80ca
# ╟─808897e2-6bb0-11eb-2865-a533dcf49b12
# ╟─8e6a1872-6bb0-11eb-3825-098258ba2745
# ╟─b5e0b0a0-6bb0-11eb-23e1-b1dd987b548f
# ╟─d7c4a62c-6bb0-11eb-39b1-8541451d8840
# ╟─ed34ff98-6bb0-11eb-14e4-71d769deba60
# ╠═1308914e-6bb1-11eb-069a-ab093d004789
# ╟─7625367c-6bb1-11eb-1db9-49460415eb3e
# ╠═7e7e2fce-6bb1-11eb-3054-d329793998dd
# ╠═e20e3002-6bbb-11eb-201e-7565164f9f11
