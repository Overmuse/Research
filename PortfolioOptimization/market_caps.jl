# June 2020
# https://www.ftserussell.com/analytics/factsheets/home/search
const MARKET_CAPS = Dict(
    :US_EQUITY        => 29_987_883,
    :DEVELOPED_EQUITY => 17_287_702,
    :EMERGING_EQUITY  =>  5_679_935, #
    :US_IG_DEBT       => 23_471_030, # USBIG
    :US_HY_DEBT       =>  1_131_740, # HYM
    :DEVELOPED_DEBT   => 20_842_880, # WorldBIG - USBIG
    :EMERGING_DEBT    =>  2_113_490, # EMUSDBBI
)

function Base.map(f::Function, d::Dict{K, V}) where {K, V}
    Dict(
        map(zip(keys(d), values(d))) do (k, v)
            k => f(v)
        end
    )
end
