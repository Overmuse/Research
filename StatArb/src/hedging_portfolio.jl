#' The OLS optimal hedging betas for the asset with *price* series y, given a matrix of 
#' price series X
function hedging_betas(X, y; λ = 0)
    inv(X' * X .+ λ * I(size(X, 2))) * X' * y
end
