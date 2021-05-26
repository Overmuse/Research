function variance(θ, δ, A, B, C, D, E, F)
    a = exp(-θ*δ)
    √(2/(77*30)*(θ/(1-a^2)*(A*a^2+B*a+C) - θ/(1-a^156)*(D*a^156+E*a^78+F)))
end

function log_likelihood(Y, L, δ, A, B, C, D, E, F, θ)
    a = exp(-θ*δ)
    σ = variance(θ, δ, A, B, C, D, E, F)
    -77 * 30log(σ)-77*30/2*log((1-a^2)/θ)-77*30/2*log(π)+30/2*log((1-a^156)/(1-a^2))-77*30/2
end

function calibrate(Y, δ₁)
    try
        L = zeros(60)
        for i in 1:30
            L[2i-1] = Y[79*(i-1)+1]
            L[2i] = Y[79*i]
        end
        L = ShiftedArray(vcat(0, L), -1)
        δ = δ₁/78
        A = sum(sum(Y[79*(i-1)+j]^2 for j in 1:78) + 78*(L[2i-2] + L[2i-1])^2/4 - (L[2i-2] + L[2i-1])*sum(Y[79*(i-1)+j] for j in 1:78) for i in 1:30)
        B = sum(-156/4*(L[2i-2]+L[2i-1])^2 + (L[2i-2] + L[2i-1]) * sum(Y[79*(i-1)+j+1] for j in 1:78) + (L[2i-2] + L[2i-1]) * sum(Y[79*(i-1)+j] for j in 1:78) - 2 * sum(Y[79*(i-1)+j+1]*Y[79*(i-1)+j] for j in 1:78) for i in 1:30)
        C = sum(sum(Y[79*(i-1)+j+1]^2 for j in 1:78) + 78/4* (L[2i-2] + L[2i-1])^2 - (L[2i-2] + L[2i-1])* sum(Y[79*(i-1)+j+1] for j in 1:78) for i in 1:30)
        D = sum(L[2i-1]^2 / 4 + L[2i-2]^2 / 4 - L[2i-1]*L[2i-2]/2 for i in 1:30)
        E = sum(L[2i-1]^2/2 - L[2i-2]^2/2 - L[2i]*L[2i-1] + L[2i]*L[2i-2] for i in 1:30)
        F = sum(L[2i]^2 + L[2i-1]^2 / 4 + L[2i-2]^2 / 4 - L[2i]*L[2i-1] - L[2i]*L[2i-2] + L[2i-1]*L[2i-2]/2 for i in 1:30)
        opt = θ -> -log_likelihood(Y, L, δ, A, B, C, D, E, F, only(θ))
        init = [0.01]
        solve = optimize(opt, init, BFGS())
        θ = only(minimizer(solve))
        σ = variance(θ, δ, A, B, C, D, E, F)
        (θ, σ)
    catch e
        (-Inf, -Inf)
    end
end

function score(θ, σ, δ₁)
    σ / (2θ) * (1 - exp(-2θ * δ₁/78))
end
