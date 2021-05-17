using NonNegLeastSquaresMLJInterface: NonNegativeLeastSquareRegressor
using MLJBase
using Random 
using Test


@testset "Testing NonNegativeLeastSquareRegressor MLJ Interface" begin
    Random.seed!(123)
    n = 100
    X = randn(n, 3)
    y = X * [2, -10, 7] + randn(n)
    model = NonNegativeLeastSquareRegressor(;alg=:nnls)
    mach = machine(model, X, y)

    @testset " With fit_intercept" begin
        # Testing fit and fitted_params method
        fit!(mach)
        fp = fitted_params(mach)

        @test all(fp.coefs .>= 0)
        @test fp.intercept >= 0

        # testing predict method
        ypred = predict(mach)

        mse = sum((ypred - y).^2) / n
        avg_mse = sum((y .- mean(y)).^2) / n

        @test mse < avg_mse
    end

    @testset " Without fit_intercept" begin
        model.fit_intercept = false
        # Testing fit and fitted_params method
        fit!(mach)
        fp = fitted_params(mach)

        @test all(fp.coefs .>= 0)
        @test fp.intercept === nothing

        # testing predict method
        ypred = predict(mach)

        mse = sum((ypred - y).^2) / n
        avg_mse = sum((y .- mean(y)).^2) / n

        @test mse < avg_mse
    end

end