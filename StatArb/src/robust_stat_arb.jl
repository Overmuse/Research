# https://www.researchgate.net/publication/335464341_Robust_Statistical_Arbitrage_Strategies

function stat_arb_bounds_fixp(x, y, prob; f = (x, y) -> abs(x - y), stat_arb=true, min=false, S₀=1, ϵ=0)
    n₁, n₂ = length(x), length(y)
    r = zeros(n₁ + 3) # RHS of the matrix
    A = zeros(n₁ + 3, n₁ * n₂) # LHS Vector

    # Martingale conditions
    for i in 1:n₁
        a = zeros(n₁, n₂)
        for j in 1:n₂
            a[i, j] = y[j] - x[i]
        end
        A[i, :] = a
    end
    
    # Martingale constraint at 0
    a = zeros(n₁, n₂)
    for i in 1:n₁, j in 1:n₂
        a[i, j] = x[i]
    end

    # measure constraint
    A[n₁+1, :] = a
    r[n₁+2] = S₀

    # not equivalent
    A[n₁ + 3, :] .= iszero(prob)
    r[n₁+ 3] = 0.0

    if stat_arb
        for i in 1:n₁, j in 1:n₂, k in 1:n₁
            if k != i
                a = zeros(n₁, n₂)
                a[i, j] = prob[k, j]
                a[k, j] = -prob[i, j]
                A = vcat(A, vec(a)')
                r = vcat(r, 0)
            end
        end
    end

    n_nonzero = 0

    # ensuring non-zero entries
    for i in 1:(n₁*n₂)
        if vec(prob)[i] > 0
            a = zeros(n₁ * n₂)
            a[i] = 1
            A = vcat(A, a')
            n_nonzero += 1 
            r = vcat(r, ϵ)
        end
    end

    # cost function
    costs = zeros(n₁, n₂)
    if min
        sense = "max"
    else
        sense = "min"
    end
    return A, costs, r
    opt<-gurobi( list(A=A,obj=costs,lb=rep(0,n1*n2),ub=rep(1,n1*n2),modelsense=sense,rhs=r,sense=c(rep("=",dim(A)[[1]]-nr_nonzero),rep(">",nr_nonzero))), params=  list( OutputFlag=0))
    
    q<-array(opt$x,dim=c(n1,n2))
    price<-opt$objval


    return(list(Q=q,Price=price))
end

function gen_data(n, N, S₀)
    u = fill(1.1, N)
    d = 1 ./ u
    p = range(0.4, 0.6, length = N)

    x = zeros(N, n÷2 + 1)
    y = zeros(N, n+1)
    prob = zeros(N, n÷2 + 1, n+1)

    for i in 1:N
        dist= Binomial(n÷2, p[i])
        for l in 0:(n÷2)
            x[i, l+1] = S₀ * u[i]^l * d[i]^(n÷2-l)
        end
        for l in 0:n
            y[i, l+1] = S₀ * u[i]^l * d[i] ^ (n-l)
        end
        for j in 1:(n÷2 + 1), k in 0:(n÷2)
            prob[i, j, k+j] = pdf(dist, j-1) * pdf(dist, k)
        end
    end
    x, y, prob
end
