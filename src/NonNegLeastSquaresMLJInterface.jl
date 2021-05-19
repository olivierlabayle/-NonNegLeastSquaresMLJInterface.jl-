module NonNegLeastSquaresMLJInterface

using MLJModelInterface
using NonNegLeastSquares: nonneg_lsq


################################
########### Structure ##########
################################


MLJModelInterface.@mlj_model mutable struct NonNegativeLeastSquareRegressor <: MLJModelInterface.Deterministic
    alg::Symbol = :pivot::(_ in (:pivot, :nnls, :fnnls, :admm))
    variant::Symbol = :none::(_ in (:comb, :cache, :none))
    gram::Bool = false
    fit_intercept::Bool = true
end


################################
########### Methods ############
################################

"""
    augment_X(X, b)
Augment the matrix `X` with a column of ones if the intercept is to be fitted (`b=true`), return
`X` otherwise.
"""
function augment_X(X::Matrix, b::Bool)::Matrix
    b && return hcat(X, ones(eltype(X), size(X, 1), 1))
    return X
end


function MLJModelInterface.fit(m::NonNegativeLeastSquareRegressor, verbosity::Int, X, y)
    X = augment_X(MLJModelInterface.matrix(X), m.fit_intercept)
    fitresult = vec(nonneg_lsq(X, y; 
                                alg=m.alg,
                                variant=m.variant,
                                gram=m.gram))
    cache = nothing
    report = NamedTuple{}()
    return (fitresult, cache, report)
end


function MLJModelInterface.fitted_params(model::NonNegativeLeastSquareRegressor, fitresult)
    return (coefs      = fitresult[1:end-Int(model.fit_intercept)],
            intercept = ifelse(model.fit_intercept, fitresult[end], nothing))
end


function MLJModelInterface.predict(m::NonNegativeLeastSquareRegressor, fitresult, Xnew)
    Xnew = augment_X(MLJModelInterface.matrix(Xnew), m.fit_intercept)
    ypred = Xnew * fitresult
    return ypred
end


################################
########### METADATA ###########
################################

const ALL_MODELS = Union{NonNegativeLeastSquareRegressor}


MLJModelInterface.metadata_pkg.(ALL_MODELS,
    name       = "NonNegLeastSquares",
    uuid       = "b7351bd1-99d9-5c5d-8786-f205a815c4d",
    url        = "https://github.com/ahwillia/NonNegLeastSquares.jl",
    julia      = true,
    license    = "MIT",
    is_wrapper = false,
)


MLJModelInterface.metadata_model(NonNegativeLeastSquareRegressor,
    input_scitype    = MLJModelInterface.Table(MLJModelInterface.Continuous),
    target_scitype   = AbstractVector{MLJModelInterface.Continuous},
    output_scitype   = MLJModelInterface.Table(MLJModelInterface.Continuous),
    supports_weights = false,
    descr            = "A non negative least square regressor model",
	load_path        = "NonNegLeastSquaresMLJInterface.NonNegativeLeastSquareRegressor"
    )


end