function intraday_returns(L)
    L[2:2:end] .- L[1:2:end]
end

function overnight_returns(L)
    L[3:2:end] .- L[2:2:end-2]
end

function initialize_deltas(L)
    intraday = intraday_returns(L)
    overnight = overnight_returns(L)
    ratio = var(intraday) / var(overnight)
    δ₂ = 1/(252 * (ratio + 1))
    δ₁ = 1/252 - δ₂
    (δ₁, δ₂)
end

function log_likelihood(L, δ₁, δ₂, θ, σ₁, σ₂)
    (σ₁ < 0 || σ₂ < 0) && return -Inf
    N = length(L)÷2
    -N / 2 * log(2π) - N * log(σ₁) - 1/(2σ₁^2)*sum((L[2i]-L[2i-1]*exp(-θ*δ₁))^2 for i in 1:N) - (N-1)/2 * log(2π) - (N-1)*log(σ₂) - 1/(2σ₂^2)*sum((L[2i+1]-L[2i]*exp(-θ*δ₂))^2 for i in 1:N-1)
end

function refine_deltas(L, θ, δ₁)
    f₁ = δ₁ -> var(L[2:2:end] .- L[1:2:end]*exp(-θ*δ₁)) / var(L[3:2:end] .- L[2:2:end-2]*exp(-θ*(1/252 - δ₁))) - (1-exp(-2θ*δ₁))/(1-exp(-2θ*(1/252-δ₁)))
    δ₁ = find_zero(f₁, δ₁)
    δ₂ = 1/252 - δ₁
    δ₁, δ₂
end

function calibrate_L(L)
    intraday = intraday_returns(L)
    overnight = overnight_returns(L)
    tol = Inf
    δ₁, δ₂ = initialize_deltas(L)
    θ, σ₁, σ₂ = 0.0, 0.0, 0.0
    iterations = 0
    while tol > 1e-6
        if iterations > 50
            @warn "Max iterations reached"
            break
        end
        opt = params -> -log_likelihood(L, δ₁, δ₂, params...)
        init = [0.0, 0.1, 0.1]
        lower = [-Inf, 0.0, 0.0]
        upper = [Inf, Inf, Inf]
        solve = optimize(opt, lower, upper, init, Fminbox(BFGS()))
        θ, σ₁, σ₂ = minimizer(solve)
        δ₁_temp, δ₂_temp = refine_deltas(L, θ, δ₁)
        tol = abs(δ₁ - δ₁_temp) + abs(δ₂ - δ₂_temp)
        δ₁, δ₂ = δ₁_temp, δ₂_temp
        iterations += 1
    end
    σ = σ₁ / √((1-exp(-2θ*δ₁))/(2θ))
    θ, δ₁, σ
end

function L_score(θ, σ)
    θ < 0 && return Inf
    σ / (2θ) * (1 - exp(-2θ*(1/252)))
end
    
