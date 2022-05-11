# following david anthoff 

using VegaLite, VegaDatasets, Query, DataFrames

cars = dataset("cars")

cars |> @select(:Acceleration,:Name) |> collect

function foo(data,origin)
    df = data |> @filter(_.Origin == origin) |> DataFrame
    return df |> @vlplot(:point,:Acceleration,:Miles_per_Gallon)
end

p=foo(cars,"USA")

p |> save("foo.png")

using PyCall

using PyCall

math = pyimport("math")

math.sin( π/2)

so = pyimport("scipy.optimize")
so.newton(x -> cos(x) - x, 1)

py"""
import numpy as np

def sinpi(x):
    return np.sin(np.pi * x)
"""
py"sinpi"(1)
