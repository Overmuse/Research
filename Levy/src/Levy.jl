module Levy

struct LevyProcess1
    θ::Float64
end

function fit(LevyProcess, X, μ = 0)
    n = length(X)
    Δₙ = 1 / (252*391)
    β = 0.5 - eps()
    νₙ = Δₙ^β
    ΔX = X[2:end] .- X[1:end-1]
    𝟙 = abs.(ΔX) .<= νₙ
    numerator = sum((μ - X[i])*ΔX'𝟙 for i in 1:n-1)
    denominator = sum((μ - X[i])^2*(Δₙ) for i in 1:n-1)
    numerator / denominator
end

const LevyProcess = LevyProcess1

function regime_classification(X)
    p = fit(LevyProcess, X)
    bic = calculate_BIC(X, p)
    r = 2
    while true
        I = start_grid(X, r)
        l = num_start(X, r)
        a = 1
        bic[a] = Inf
        c_b = zeros(r-1)
        c_best = zeros(r-1)
        while true
            i = 1
            while true
                c = I[i]
                S = classify_data(X, r, c)
                p = fit.(Ref(LevyProcess), S)
                bic_local = calculate_BIC.(S, p)
                if i == z
                    break
                end
                i += 1
            end
            a += 1
            c_bprev = c_b
            bic[a] = minimum(bic_local)
            c_b = I[argmin(bic_local)]
            if bic[a-1] < bic[a] || a-1 == l
                break
            end
            I = smart_grid(X, I, bic_local)
        end
        c_bestprev = c_best
        c_best = c_bprev
        bic⋆[r] = bic[a-1]
        if bic⋆[r-1] < bic⋆
            break
        end
        r += 1
    end
    S = classify_data(X, r-1, c_best_prev)
    p = fit.(Ref(LevyProcess), S)
    bic = calculate_BIC.(S, p)
end


end # module
