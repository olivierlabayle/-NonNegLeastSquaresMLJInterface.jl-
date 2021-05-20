![CI](https://github.com/olivierlabayle/NonNegLeastSquaresMLJInterface.jl/actions/workflows/main.yml/badge.svg)
[![codecov](https://codecov.io/gh/olivierlabayle/NonNegLeastSquaresMLJInterface.jl/branch/main/graph/badge.svg?token=L6I102JZ9L)](https://codecov.io/gh/olivierlabayle/NonNegLeastSquaresMLJInterface.jl)
[![GitHub license](https://img.shields.io/github/license/olivierlabayle/NonNegLeastSquaresMLJInterface.jl)](https://github.com/olivierlabayle/NonNegLeastSquaresMLJInterface.jl/blob/main/LICENSE)

# NonNegLeastSquaresMLJInterface.jl

A MLJ Interface to the [NonNegLeastSquares.jl](https://github.com/ahwillia/NonNegLeastSquares.jl) package.

## Installation

```julia
add NonNegLeastSquaresMLJInterface
```

## Usage

```julia
using NonNegLeastSquaresMLJInterface: NonNegativeLeastSquareRegressor

n = 100
X = randn(n, 3)
y = X * [2, -10, 7] + randn(n)

model = NonNegativeLeastSquareRegressor(;alg=:nnls)
mach = machine(model, X, y)

fit!(mach)
fp = fitted_params(mach)
```
