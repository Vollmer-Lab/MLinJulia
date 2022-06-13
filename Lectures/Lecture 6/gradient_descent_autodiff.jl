### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ e03882f9-843e-4552-90b1-c47b6cbba19b
using PlutoUI; PlutoUI.TableOfContents()

# ╔═╡ 7aa547f8-25d4-488d-9fc3-f633f7f03f57
begin
    #using Pkg
	#Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using MLJ, MLJLinearModels, Plots, LinearAlgebra, Flux, Random, DataFrames, Distributions#, MLCourse
    #import MLCourse: poly
end

# ╔═╡ a723876e-ae49-4b3d-8cd6-d93efb774ef5
using BenchmarkTools

# ╔═╡ de2ea96e-df6f-4799-b912-07de7e5b965a
using JuMP, Ipopt

# ╔═╡ 28408f30-dfa4-4272-a215-16812eaa4398
begin
	polysymbol(x, d) = Symbol(d == 0 ? "" : d == 1 ? "$x" : "$x^$d")
	
	colnames(df::AbstractDataFrame) = names(df)
	colnames(d::NamedTuple) = keys(d)
	colname(names, predictor::Int) = names[predictor]
	colname(names, predictor::Symbol) = predictor ∈ names || string(predictor) ∈ names ? Symbol(predictor) : error("Predictor $predictor not found in $names.")
	function poly(data, degree, predictors::NTuple{1} = (1,))
	    cn = colnames(data)
	    col = colname(cn, predictors[1])
	    res = DataFrame([getproperty(data, col) .^ k for k in 1:degree],
	                    [polysymbol(col, k) for k in 1:degree])
	    if hasproperty(data, :y)
	        res.y = data.y
	    end
	    res
	end
	function poly(data, degree, predictors::NTuple{2})
	    cn = colnames(data)
	    col1 = colname(cn, predictors[1])
	    col2 = colname(cn, predictors[2])
	    res = DataFrame([getproperty(data, col1) .^ d1 .* getproperty(data, col2) .^ d2
	                     for d1 in 0:degree, d2 in 0:degree if 0 < d1 + d2 ≤ degree],
	                    [Symbol(polysymbol(col1, d1), polysymbol(col2, d2))
	                     for d1 in 0:degree, d2 in 0:degree if 0 < d1 + d2 ≤ degree])
	    if hasproperty(data, :y)
	        res.y = data.y
	    end
	    res
	end
	poly(_, _, _) = error("Polynomials in more than 2 predictors are not implemented.")
end

# ╔═╡ 9f18f610-79e2-403a-a792-fe2dafd2054a
md"# Automatic Differentiation and Gradient Descent

## Automatic Differentiation

We often have functions we want to take the derivative/gradient/Jacobian/Hessian of for various reasons:

Derivatives help us optimize functions. 
One common way to calculate derivatives is simply to take an analytic derivative. For example, let's look at the following function $h(x)$.

$$h(x) = \sin(x^3)$$


If we want $\frac{dh(x)}{dx}$ we can calculate this with the chain rule:

$$\frac{dh(x)}{dx} = \frac{dsin(x^2)}{dx} = \cos(x^2)\frac{dx^2}{dx} = 2x\cos(x^2)$$

In Julia, this would be written up as
"

# ╔═╡ 2803f283-4800-47d5-8822-f7a45d945c3e
let
	h(x) = sin(x^2)
	dh(x) = 2 * x * cos(x^2)
	# 100 values of x from 0,3
	xs = range(0, 3, length=100)
	plot(xs, h, label="h(x)")
	plot!(xs, dh, label="dh(x)")
end

# ╔═╡ 7cf19059-f00d-4e04-a7fe-41274f3c0d8a
md"""
Sometimes, however, it's a massive pain to calculate analytic derivatives. Maybe you have a lot of parameters to keep track of.

Another approach to calculating derivatives is the use of finite differences, 
"
$$\frac{dh}{dx} = \frac{h(x + \Delta)}{\Delta}$$

but this technique has 2 major downsides. The first is that the stability of this technique depends on the step size $\Delta$. The second is that for a multivariable function $f:\mathcal{R^n} \rightarrow \mathcal{R}$ using this technique would require $\mathcal{O}(N)$ function evaluations, 

$$\frac{\partial f}{\partial x_i} = \frac{f(x_1, ..., x_i + \Delta, ..., x_n)-f(x_1, ..., x_i , ..., x_n)}{\Delta}$$

to compute the gradient vector $\big <\frac{\partial f}{\partial x_1}, ..., \frac{\partial f}{\partial x_n}\big>$ .

"""

# ╔═╡ 7630772c-c505-4ea3-a7a8-9cf54aee189a
import FiniteDifferences

# ╔═╡ 9d04de9a-6220-4a16-be52-7714f21f193b
let
	h(x) = sin(x^2)
	dh(x) = 2 * x * cos(x^2)
	dh_central_fdm(x) = FiniteDifferences.central_fdm(2, 1)(h, x); #Second order central method.
	# 100 values of x from 0,3
	xs = range(0, 3, length=100)
	plot(xs, h, label="h(x)")
	plot!(xs, dh, label="dh(x)")
	plot!(xs, dh_central_fdm, label="dh(x) central difference")
end

# ╔═╡ a976b898-37ba-430a-b8b1-2b6c55dfd0c6
md"""
Yet another approach is to use Symbolic integration on the computer. This is a good solution, if the symbolic package can handle your functions. Doing algebra by hand is tedious and error-prone, but is sometimes invaluable.
"""

# ╔═╡ 3ea9881c-91c1-4aac-a2ab-4097e52d9ddf
md"""
Enter automatic/algorithmic differentiation. This is how we can get $dh(x)$ without doing boring old normal math!

### Automatic differentiation (autodiff)
Automatic differentiation is simply a programmatic way of handling the chain rule. There are two "modes" of autodiff, called "reverse mode" and "forward mode".

Both modes yield the same derivative, but get there different ways.

Forward mode calculates derivatives from the inside out
Reverse mode calculates derivatives from the outside in
I will talk later about which is "better", but for right now the gist is that forward mode is good for small numbers of parameters, and reverse mode is good for large numbers of parameters (for scalar functions).

We will study how forward diff works in detail later.

I just want you to limp along for now, and going through the details of AD is not particularly important until you start writing your own AD rules.

There's oodles of resources on AD and how they work, but for our purposes I just want you to understand how to hook them up.

A good resource is the ChainRules.jl documentation.

Let's do some tinkering around. Please install ReverseDiff.jl, ForwardDiff.jl, and Zygote.jl using
```
] add ReverseDiff ForwardDiff, Zygote
```
or
```
import Pkg
Pkg.add(["ReverseDiff", "ForwardDiff", "Zygote"])
```

Once those are installed, let's tinker around with how they work.

Let's only import them -- don't call using for now since the packages export similarly named functions.

For ForwardMode automatic differentian we use ForwardDiff.jl,  `ForwardDiff.derivative(f, x)`.

For ReverseMode automatic differentiation we can either use
- ReverseDiff.jl `ReverseDiff.gradient(f, x)`. (Note: ReverseDiff.jl only works with array inputs.) 
- Zygote.jl `Zygote.gradient(f, x)`

"""

# ╔═╡ 8a660a1f-b84b-4ef9-b5fb-2f1740b354d3
import ForwardDiff, ReverseDiff, Zygote

# ╔═╡ 196c7318-3b97-4ea4-a80d-17d2e9567193
let
	f(x) = sin(x^2)
	
	exact_df(x) = 2*x*cos(x^2)
	forward_df(x) = ForwardDiff.derivative(f, x)
	forward_grad_df(x) = ForwardDiff.gradient(z -> f(z[1]), [x])[1]
	reverse_df(x) = ReverseDiff.gradient(z -> f(z[1]), [x])[1]
	zygote_reverse_df(x) = Zygote.gradient(f, x)[1]
	
	@info "AD Examples" exact_df(1.0) reverse_df(1.0) forward_df(1.0) forward_grad_df(1.0) zygote_reverse_df(1.0) 

	# plot the functions
	xs = range(0, 3, length=100)
	plot(xs, f, label="f(x)")
	plot!(xs, exact_df, label="exact_df(x)")
	plot!(xs, forward_df, label="forward_df(x)")
	plot!(xs, reverse_df, label="reverse_df(x)")
	plot!(xs, zygote_reverse_df, label="zygote_reverse_df(x)", legend=:topleft)
end

# ╔═╡ bcc646a2-287d-48db-95f5-1ebf6f0d5f4c
md"""
#### Concept check!
Let's expand what we've worked with into general multivariate functions.

Take the loss function for OLS, as an example.

$$f(\beta, x, y) = (Y - X\beta)'(Y - X\beta)$$

Use ForwardMode and/or ReverseMode automatic differentiation to calculate the gradient w.r.t. $\beta$, $\nabla f(\cdot)$, at $\hat{\beta} = (X'X)^{-1}X'Y$ 
. This gradient should be approximately zero for both, since this is the exact analytic value that minimizes the gradient.

"""

# ╔═╡ 2a0bdd82-2de8-4990-9357-52c8270e592d
let
	beta = [1.0, 2.2] # population parameters.

	# Generate noisy Data to perform Least Squares on.
	n = 100
	X = [ones(n) rand(n)]
	Y = X*beta + randn(n)

	beta_hat = inv(X'X)*X'Y #Least Squares Solution gotten Analytically.
	
	f(b) = (Y - X*b)' * (Y - X*b) #OLS Loss function.
	df(b) = ReverseDiff.gradient(f, b) # Derivative of loss function.

	# evaluating the derivative of OLS loss function at `beta_hat` should give the zero vector.
	beta_hat, df(beta_hat)
end

# ╔═╡ b5f0e109-ac66-40f7-9962-aa89b8b5de65
md"""
In the following sections **AD** refers to techniques implemented in ForwardDiff.jl and ReverseDiff.jl. Zygote.jl would be discussed in a later section.
### AD (usually) works on any function
A really cool thing is you can write really gnarly functions and get their derivatives easily and quickly. For example:
"""

# ╔═╡ b4039c01-ddee-4818-ae42-802eee4fa15a
let
	choosy(arr) = choosy(arr...)
function choosy(x, μ, σ)
    if x >= 1
        return logpdf(LogNormal(μ, σ), x)
    elseif x <= -1
        return logpdf(Normal(μ, σ), x) 
    else
        return logpdf(Beta(15, 12), abs(x)) * sin(μ) * cos(σ)
    end
end

println(ForwardDiff.gradient(choosy, [1.0, 2.0, 3.0]))
println(ForwardDiff.gradient(choosy, [0.5, 2.0, 3.0]))
println(ForwardDiff.gradient(choosy, [-1.2, 2.0, 3.0]))
end

# ╔═╡ 4a53fc66-6822-4f85-81a7-319551e4882b
md"""
### AD woes
The way that our two AD packages work is that they change the underlying type of your input.

What does this mean?

**Do not make your functions too type constricted if you want to get their derivative!**

Let me show you what I mean.
"""

# ╔═╡ b08dffa2-686c-4651-8e06-ee2362ead7c3
begin
	function g_basic(x::Float64)
	    return cos(sqrt(abs(sin(x^2))))
	end
	
	local xs = range(0, 3, length=100)
	plot(xs, g_basic)
end

# ╔═╡ 219399db-4ab5-43fe-bab4-a24fc2608f72
ForwardDiff.derivative(g_basic, 1.0)

# ╔═╡ 9c2c5079-9357-4765-bf61-2db9b096e667
md"""
What happened here?

Well, when I wrote `g`, I constricted the input to be `Float64`. This is fine when I call `g(1.0)`, but when I run

`ForwardDiff.derivative(g, 1.0)`

ForwardDiff wraps the input value `1.0` in what is called a dual. A dual value accumulates the derivatives from the inside of a function and carries them all the way out.

Let's see how to fix it.
"""

# ╔═╡ 1137bb34-e4d2-4344-9f85-8792158e505c
let
	function g(x) # Note that I have removed the type constraint! Let multiple dispatch work for you.
	    return cos(sqrt(abs(sin(x^2))))
	end

	ForwardDiff.derivative(g, 1.0)
end

# ╔═╡ 96a5358f-c0c5-4c87-893d-812075a6f983
md"""
Pro tip: when working with automatic differentiation, only annotate types when it is strictly necessary!

If you absolutely must add type annotations because the compiler cannot infer which function to dispatch to, it is best to only constraint your types as far as `Real`, i.e.
"""

# ╔═╡ fc1d9f67-140d-4484-af75-d6a7996ccb2d
let
	function g(x::Real) # Using a ::Real type constraint
    	return cos(sqrt(abs(sin(x^2))))
	end

	ForwardDiff.derivative(g, 1.0)
end

# ╔═╡ b2ae0ab8-42df-4e1f-890a-b321fd8d71ea
md"ReverseDiff does the exact same thing, though with a different type:"

# ╔═╡ 35e51599-b306-43b1-9139-0431a70ec88c
let
	function g_rev(x::Float64) # Using a ::Float64 type constraint
		println(typeof(x))
		return cos(sqrt(abs(sin(x^2))))
	end

	@show ReverseDiff.gradient(g_rev, 1.0)[1]
end

# ╔═╡ 70dabe04-94cc-4900-a893-2131fba389ed
md"""
### In-place derivatives
A big computational cost can be allocating memory to store gradients in arrays.

In Julia, it's common to provide a buffer to store gradients in. Buffers are particularly useful if you end up computing a lot of gradients.

From here on out I'm going to use gradient, because it's much more common and accessible than derivative.
"""

# ╔═╡ 119d8b81-96a2-4e62-aac3-b7df5bf77ee2
begin
	function fun_function(p)
	    return sin(p[1]^2) * cos(p[2] - p[1]) + p[3]^3
	end
	
	# Version that allocates a return value each time
	result = ForwardDiff.gradient(fun_function, [1.0, 2.1, 2.1])
	
	# Version that places the result in an existing array
	buf = zeros(3)
	ForwardDiff.gradient!(buf, fun_function, [1.0, 2.1, 2.1])
	
	# Verify that the results are the same.
	buf == result
end

# ╔═╡ 2022c744-a0ab-4817-a45b-7fe94360451c
md"""
This is **really** valuable for performance if you run a lot of derivatives.

Let's say for example that we wanted the gradient of this function across a grid of `p`
"""

# ╔═╡ 0ae02b75-435a-4ef3-8fdf-5422653c84cd
begin
	rng_ = range(-5, 5, length=100)
	p_grid = vec(collect([a,b,c] for a in rng_, b in rng_, c in rng_));
end

# ╔═╡ 33c8744d-1205-46e2-93a7-85ef35efb081
p_grid[1:3]

# ╔═╡ d420f077-d0a5-4361-b23f-bfb5a8ff450e
begin
	# Using ForwardDiff.gradient
	function simple_grad(p_grid)
	    grads = zeros(length(p_grid), 3)
	    for i in 1:length(p_grid)
	        grads[i,:] = ForwardDiff.gradient(fun_function, p_grid[i])
	    end
	    return grads
	end
end

# ╔═╡ 4274811c-a361-440a-9f8d-1bdb607ebbf5
begin
	# Using ForwardDiff.gradient! with a buffer
	
	function buf_grad(p_grid)
	    grads = zeros(length(p_grid), 3)
	    for i in 1:length(p_grid)
	        # Note that I use view() here -- this gives the LOCATION of an 
	        # array. Using grads[i,:] copies the array.
	        ForwardDiff.gradient!(view(grads, i,:), fun_function, p_grid[i])
	    end
	    return grads
	end
	
end

# ╔═╡ 7706d51c-7244-4077-96b3-4432a8093e59
md"""
`buf_grad` is way faster that `simple_grad` -- it's got about half the amount of allocations.
"""

# ╔═╡ 8764c694-5fd6-49ea-aa08-8792f00e7593
@benchmark(simple_grad($p_grid))

# ╔═╡ f595bf3e-c063-4d29-9123-ad62a8a77e97
@benchmark buf_grad($p_grid)

# ╔═╡ 791b26ad-f259-4cd0-afc1-b5c658ad789d
md"""
### Reverse mode or forward mode?
People generally recommend ForwardDiff for gradients with less than 100 parameters, and ReverseDiff for more than 100 parameters.

Of course, they also differ a little in implementation. ReverseDiff allows you to precompile a lot of your gradient computation so you can reuse parts of the gradient.

Lets play around a little so you can see when ReverseDiff is preferred to ForwardDiff.
"""

# ╔═╡ 16cca6d5-d933-4203-b8a5-6049ad00a3b0
begin
	# This function has a variable parameter size p
	# How slow/fast is it to use different AD methods?
	function big_function(p)
	    n = div(length(p), 2)
	    alphas = p[1:n]
	    betas = p[n+1:end]
	    
	    return sum(alphas .* sin.(cos.(betas)))
	end
	
	xs_2 = randn(2)
	xs_10 = randn(10)
	xs_1k = randn(1000)
	xs_10k = randn(10_000);
end

# ╔═╡ 96ff3a8b-2aa9-476c-b3d9-6dc1cabbac36
@benchmark ForwardDiff.gradient(big_function, xs_2) # 2 parameters

# ╔═╡ c4c855d0-745a-4519-8448-38e5993df5df
@benchmark ReverseDiff.gradient(big_function, xs_2) # 2 parameters

# ╔═╡ d9b0e98f-d12a-4208-a902-f9c31f93ce91
@benchmark ForwardDiff.gradient(big_function, xs_10) # 10 parameters

# ╔═╡ c780c4c5-50e3-4823-883b-87e2d33b34b5
@benchmark ReverseDiff.gradient(big_function, xs_10) # 10 parameters

# ╔═╡ 250a0377-cd24-4756-8b2e-da1f432c86b4
@benchmark ForwardDiff.gradient(big_function, xs_1k) # 1k parameters

# ╔═╡ adbcd6cb-3929-4ee0-ba74-e452e753572d
@benchmark ReverseDiff.gradient(big_function, xs_1k)# 1k parameters

# ╔═╡ 677327cf-628f-446c-be22-b3ec8c89a785
@benchmark ForwardDiff.gradient(big_function, xs_10k) # 10k parameters

# ╔═╡ 0e73bf21-b241-4589-a532-f8a182b826ff
@benchmark ReverseDiff.gradient(big_function, xs_10k) # 10k parameters

# ╔═╡ 134b91d4-a26c-430a-bc3f-0cb41feb531e
md"""
**Conclusion! ReverseDiff for bigger models!**

ForwardDiff is often faster than ReverseDiff for lower dimensional gradients (length(input) < 100), or gradients of functions where the number of input parameters is small compared to the number of operations performed on them. ReverseDiff is often faster if your code is expressed as a series of array operations, e.g. a composition of Julia's Base linear algebra methods.
"""

# ╔═╡ 526192ca-5ed7-4e4b-b05c-53d706bcc032
md"""
### Zygote.jl

"""

# ╔═╡ d2d46434-b6e8-4685-8960-0024db9c8539
md"""
## Gradient Descent
### Gradient Descent for Linear Regression
"""

# ╔═╡ fcbac567-36db-40c5-9436-6315c0caa40a
function gradient_descent(f, x, η, T; callback = x -> nothing)
    for t in 1:T
        x .-= η * gradient(f, x)[1] # update parameters in direction of -∇f
        callback(x) # the callback will be used to save intermediate values
    end
    x
end

# ╔═╡ 49e744dc-33b7-455a-9cf7-4e8e12f11152
X, y = make_regression(40, 1, rng = 16) # make_regression is an MLJ data generator

# ╔═╡ dcc76840-6552-4a86-8268-1291ab8ea0d3
begin
    lin_reg_loss(β₀, β₁, x, y) = mean((y .- β₀  .- β₁ * x).^2)
    lin_reg_loss(β₀, β₁) = lin_reg_loss(β₀, β₁, X.x1, y)
    lin_reg_loss(β) = lin_reg_loss(β[1], β[2])
end

# ╔═╡ 3ca1e8f3-d69f-454f-b917-4bbe2dcfce01
md"β₀⁽⁰⁾ = $(@bind b0 Slider(-3.:.1:3., show_value = true, default = 0.))

β₁⁽⁰⁾ = $(@bind b1 Slider(-3.:.1:3., show_value = true, default = -3.))

η = $(@bind η Slider(.01:.01:2, show_value = true))

step t = $(@bind t Slider(0:100, show_value = true))
"

# ╔═╡ b1fc14bb-1fd2-4739-a761-7a605fd4559b
begin
    params = [b0, b1] # initial parameters
    lin_reg_path = [copy(params)] # we will copy all parameter values to this list
    gradient_descent(lin_reg_loss, params, η, 100,
                     callback = x -> push!(lin_reg_path, copy(x)))
end

# ╔═╡ bbc0b514-4789-44d1-8d90-9fc325d9ad6b
let
    p1 = scatter(X.x1, y, xlim = (-2.5, 2.5), ylim = (-2, 2),
                 legend = :bottomright,
                 xlabel = "x", ylabel = "y", label = "training data")
    plot!(x -> lin_reg_path[t+1]' * [1, x], c = :red, w = 2, label = "current fit")
    p2 = contour(-3:.1:3, -3:.1:3, lin_reg_loss, cbar = false, aspect_ratio = 1)
    scatter!(first.(lin_reg_path), last.(lin_reg_path), markersize = 1, c = :red, label = nothing)
    scatter!([lin_reg_path[t+1][1]], [lin_reg_path[t+1][2]], label = nothing,
             markersize = 4, c = :red, xlabel = "β₀", ylabel = "β₁")
    plot(p1, p2, layout = (1, 2), size = (700, 400))
end


# ╔═╡ eb9e2b3f-dc4b-49a2-a611-b45f2918adcf
md"### Gradient Descent for Logistic Regression"

# ╔═╡ 402bb576-8945-403e-a9a6-fd5bfb8016bc
begin
    X2, y2 = make_regression(50, 1, binary = true, rng = 161)
	y2 = coerce(y2, MLJ.Continuous) .- 1
    σ(x) = 1/(1 + exp(-x))
    log_reg_loss(β) = log_reg_loss(β[1], β[2])
    function log_reg_loss(β₀, β₁)
        p = σ.(β₀  .+ β₁ * X2.x1) # probability of positive class for all inputs
        -mean(@. y2 * log(p) + (1-y2) * log(1 - p)) # negative log-likelihood
    end
end

# ╔═╡ cd5f079f-a06d-4d55-9666-e2b05ddf8989
md"β₀⁽⁰⁾ = $(@bind b02 Slider(-3.:.1:3., show_value = true, default = -2.))

β₁⁽⁰⁾ = $(@bind b12 Slider(-3.:.1:3., show_value = true, default = 2.))

η = $(@bind η2 Slider(.1:.1:2, show_value = true))

step t = $(@bind t2 Slider(0:100, show_value = true))
"

# ╔═╡ 93699313-5e42-48a7-abc6-ad146e5bdcdd
begin
    params2 = [b02, b12]
    log_reg_path = [copy(params2)]
    gradient_descent(log_reg_loss, params2, η2, 100,
                     callback = x -> push!(log_reg_path, copy(x)))
end

# ╔═╡ 33bdc912-e46a-4310-9184-733be7871768
let
    p1 = scatter(X2.x1, y2, xlim = (-3.5, 3.5), ylim = (-.1, 1.1),
                 legend = :bottomleft, marker_style = :vline,
                 xlabel = "x", ylabel = "y", label = "training data")
    plot!(x -> σ(log_reg_path[t2+1]' * [1, x]), c = :red, w = 2, label = "current fit")
    p2 = contour(-3:.1:3, -3:.1:3, log_reg_loss, cbar = false, aspect_ratio = 1)
    scatter!(first.(log_reg_path), last.(log_reg_path), markersize = 1, c = :red, label = nothing)
    scatter!([log_reg_path[t2+1][1]], [log_reg_path[t2+1][2]], label = nothing,
             markersize = 4, c = :red, xlabel = "β₀", ylabel = "β₁")
    plot(p1, p2, layout = (1, 2), size = (700, 400))
end


# ╔═╡ 979feb42-c328-44b5-b519-8e8cda474140
md"### Gradient Descent to Minimize a Complicated Function"

# ╔═╡ c448238c-f712-4af3-aedb-cec3a2c1a73e
begin
	Random.seed!(5)
    f(x, θ) = θ[1] * sin.(θ[2] * x .+ θ[3]) .+ θ[4] * sin.(θ[5] * x .+ θ[6])
    f(x) = 0.3 * sin(2x) + .7 * sin(3.5x + 1)
    x3 = randn(50)
    y3 = f.(x3)
end

# ╔═╡ 6e38a3f6-a592-4dc1-a6c4-d0f0050c9399
special_loss(θ) = mean((y3 .- f(x3, θ)).^2)

# ╔═╡ 4b3ac87f-39f6-4e6c-b7b3-536925e9b112
md"step t = $(@bind t3 Slider(1:10^4, show_value = true))

seed = $(@bind special_seed Slider(1:10, default = 3, show_value = true))"

# ╔═╡ 9a61332f-cdc2-4129-b7b5-5ab54ba387a3
begin
	Random.seed!(special_seed) # defined by the slider below
    params3 = .1 * randn(6)
    special_path = [copy(params3)]
    gradient_descent(special_loss, params3, .1, 10^4,
                     callback = x -> push!(special_path, copy(x)))
end

# ╔═╡ 01221937-b6d9-4dac-be90-1b8a1c5e9d87
let
    p1 = plot(f, label = "orginal function", xlabel = "x", ylabel = "y",
              ylim = (-1.3, 1.2), xlim = (-3, 3))
    plot!(x -> f(x, special_path[t3]), label = "current fit")
	scatter!(x3, y3, label = "training data")
    th = round.(special_path[t3], digits = 2)
    annotate!([(0, -1.1, text("f̂(x) = $(th[1]) * sin($(th[2])x + $(th[3])) + $(th[4]) * sin($(th[5])x + $(th[6]))", pointsize = 7))])
    losses = special_loss.(special_path)
    p2 = plot(0:10^4, losses, label = "learning curve", c = :black, yscale = :log10)
    scatter!([t3], [losses[t3]], label = "current loss", xlabel = "t", ylabel = "loss")
    p3 = contour(-4:.1:4, -4:.1:4, (x2, x5) -> special_loss([special_path[t3][1]; x2; special_path[t3][3:4]; x5; special_path[t3][6]]), xlabel = "θ₂", ylabel = "θ₅", title = "loss")
	scatter!([special_path[t3][2]], [special_path[t3][5]])
    plot(p1, plot(p2, p3, layout = (2, 1)), layout = (1, 2), size = (700, 400))
end

# ╔═╡ f5e27275-8751-4e80-9888-c3d22d8e80e3
md"*Optional* If you want to know more about the magic of automatic differentiation and how julia computes derivates of (almost) arbitrary code: have a look at this [didactic introduction to automatic differentiation](https://github.com/MikeInnes/diff-zoo) or this [video with code examples in python](https://www.youtube.com/watch?v=wG_nF1awSSY&t)."

# ╔═╡ ca1301fd-8978-44d6-bfae-e912f939d7a8
md"### Stochastic Gradient Descent

In each step of stochastic gradient descent (SGD) the gradient is computed only on a (stochastically selected) subset of the data. As long as the selected subset of the data is somewhat representative of the full dataset the gradient will point more or less in the same direction as the full gradient computed on the full dataset. The advantage of computing the gradient only on a subset of the data is that it takes much less time to compute the gradient on a small dataset than on a large dataset. In the figure below you see the niveau lines of the full loss in black and the niveau lines of the loss computed on a green, purple, yellow and a blue subset. The gradient computed on, say, the yellow subset is always perpendicular to the yellow niveau lines but it may not be perpendicular to the niveau lines of the full loss. When selecting in each step another subset we observe a jittered trajectory around the full gradient descent trajectory.
"

# ╔═╡ 75528011-05d9-47dc-a37b-e6bb6be52c25
md"η = $(@bind η_st Slider(.01:.01:.1, show_value = true))

step t = $(@bind t5 Slider(0:100, show_value = true))
"

# ╔═╡ e821fb15-0bd3-4fa7-93ea-692bf05097b5
begin
    lin_reg_loss_st = let batch = rand(1:4, 101), i = 0
        function(β)
            xb, yb = Zygote.ignore() do
                i += 1
                idxs = (batch[i]-1)*10 + 1:batch[i]*10
                X.x1[idxs], y[idxs]
            end
            lin_reg_loss(β[1], β[2], xb, yb)
        end
    end
    lin_reg_loss_b(i) = (β₀, β₁) -> lin_reg_loss(β₀, β₁, X.x1[(i-1)*10+1:i*10],
                                                 y[(i-1)*10+1:i*10])
    params_st = [b0, b1]
    lin_reg_stoch_path = [copy(params_st)]
    gradient_descent(lin_reg_loss_st, params_st, η_st, 100,
                     callback = x -> push!(lin_reg_stoch_path, copy(x)))
end;

# ╔═╡ 7541c203-f0dc-4445-9d2a-4cf16b7e912a
let
    b = lin_reg_loss_st.batch[t5+1]
    colors = [:green, :blue, :orange, :purple]
    ma = fill(.2, 40)
    ma[(b-1)*10 + 1:b*10] .= 1
    p1 = scatter(X.x1, y, xlim = (-2.5, 2.5), ylim = (-2, 2),
                 legend = false, ma = ma,
                 c = vcat([fill(c, 10)
                           for c in colors]...),
                 xlabel = "x", ylabel = "y", label = "training data")
    plot!(x -> lin_reg_stoch_path[t5+1]' * [1, x], c = :red, w = 2,
          label = "current fit")
    p2 = contour(-3:.1:3, -3:.1:3, lin_reg_loss, cbar = false,
                 linestyle = :dash, c = :black, aspect_ratio = 1)
    for i in 1:4
        contour!(-3:.1:3, -3:.1:3, lin_reg_loss_b(i), cbar = false, w = 2,
                 linestyle = :dash, c = colors[i], alpha = b == i ? 1 : .2)
    end
    plot!(first.(lin_reg_path), last.(lin_reg_path),
          c = :black, w = 3, label = "GD")
    plot!(first.(lin_reg_stoch_path), last.(lin_reg_stoch_path),
          c = :red, w = 1.5, label = "SGD")
    ps = lin_reg_stoch_path[t5+1]
    scatter!([ps[1]], [ps[2]], label = nothing, c = :red)
    plot(p1, p2, layout = (1, 2), size = (700, 400))
end


# ╔═╡ 913cf5ee-ca1e-4063-bd34-6cccd0cc548b
md"### Improved Versions of (S)GD

There are many tricks to improve over standard (stochastic) gradient descent.
One popular idea is to use [momentum](https://distill.pub/2017/momentum/).
We do not discuss these ideas further here, but you should know that `ADAM()` and `ADAMW()` are particularly popular (and successful) improvements of standard (S)GD. These methods usually require no (or very little) tuning of the learning rate."

# ╔═╡ eb289254-7167-4183-a4d0-52f68be66b04
begin
	Random.seed!(1234)
    params4 = .1 * randn(6)
    special_path2 = [copy(params4)]
	opt = ADAMW()
    for _ in 1:10^4
        Flux.Optimise.train!(_ -> special_loss(params4), Flux.params([params4], []), [nothing], opt, cb = () -> push!(special_path2, copy(params4)))
    end
end

# ╔═╡ 166472c5-c0f4-4261-a476-4c9b0f82abd6
let
    p1 = plot(f, label = "orginal function", xlabel = "x", ylabel = "y",
              ylim = (-1.3, 1.2))
    plot!(x -> f(x, special_path2[t3]), label = "current fit")
    th = round.(special_path2[t3], digits = 2)
    annotate!([(0, -1.1, text("f̂(x) = $(th[1]) * sin($(th[2])x + $(th[3])) + $(th[4]) * sin($(th[5])x + $(th[6]))", pointsize = 7))])
    losses2 = special_loss.(special_path2)
    losses = special_loss.(special_path)
    p2 = plot(0:10^4, losses2, label = "learning curve", c = :black, yscale = :log10)
    plot!(0:10^4, losses, label = "GD learning curve", c = :black, linestyle = :dot)
    scatter!([t3], [losses2[t3]], label = "current loss", xlabel = "t", ylabel = "loss", legend = :bottomleft)
    plot(p1, p2, layout = (1, 2), size = (700, 400))
end


# ╔═╡ 08a9418f-786e-4992-b1a5-04cf9060f8fe
md"### Early Stopping as Regularization"

# ╔═╡ dc57d700-2a82-4ab0-9bd2-6ce622cb0fa5
begin
    g(x) = .3 * sin(10x) + .7x
    function regression_data_generator(; n, seed = 3, rng = MersenneTwister(seed))
        x = range(0, 1, length = n)
        DataFrame(x = x, y = g.(x) .+ .1*randn(rng, n))
    end
    regression_data = regression_data_generator(n = 10, seed = 8)
	regression_valid = regression_data_generator(n = 50, seed = 123)
end;

# ╔═╡ c72dc506-fb1d-42ee-83cf-cc49753ecd4f
begin
    h_training = Array(select(poly(regression_data, 12), Not(:y)))
    h_valid = Array(select(poly(regression_valid, 12), Not(:y)))
    poly_regression_loss(θ, x, y) = mean((y .- θ[1] .- x * θ[2:end]).^2)
    poly_params = 1e-3 * randn(13)
    poly_path = [copy(poly_params)]
	poly_opt = ADAMW()
    for _ in 1:10^5
        Flux.Optimise.train!((x, y) -> poly_regression_loss(poly_params, x, y),
                             [poly_params], [(h_training, regression_data.y)],
                             poly_opt,
                             cb = () -> push!(poly_path, copy(poly_params)))
    end
end

# ╔═╡ 075f35cf-4271-4676-b9f3-e2bcf610c2d1
md"step t = $(@bind t4 Slider(0:10^3:10^5, show_value = true))"

# ╔═╡ 0d431c00-9eef-4ce4-9542-9571728d1501
let
    p1 = scatter(regression_data.x, regression_data.y,
                 label = "training data", ylims = (-.1, 1.1))
    plot!(g, label = "generator", c = :green, w = 2)
    grid = 0:.01:1
    θ = poly_path[t4 + 1]
    pred =  Array(poly((x = grid,), 12)) * θ[2:end] .+ θ[1]
    plot!(grid, pred,
          label = "fit", w = 3, c = :red, legend = :topleft)
    losses = poly_regression_loss.(poly_path, Ref(h_training), Ref(regression_data.y))
    losses_v = poly_regression_loss.(poly_path, Ref(h_valid), Ref(regression_valid.y))
    p2 = plot(0:length(poly_path)-1, losses, yscale = :log10,
              c = :blue, label = "training loss")
    scatter!([t4], [losses[t4 + 1]], c = :blue, label = nothing)
    plot!(0:length(poly_path)-1, losses_v, c = :red, label = "validation loss")
    scatter!([t4], [losses_v[t4 + 1]], c = :red, label = nothing)
    vmin, idx = findmin(losses_v)
    vline!([idx], c = :red, linestyle = :dash, label = nothing)
    hline!([vmin], c = :red, linestyle = :dash, label = nothing)
    plot(p1, p2, layout = (1, 2), size = (700, 400))
end



# ╔═╡ e0cc188c-9c8f-47f1-b1fe-afc2a578973d
md"### Exercise

Assume the noise in a linear regression setting comes from a Laplace
distribution, i.e. the conditional probability density of the response is given by
``p(Y = y | X = x, \beta) = \frac1{2s}\exp(-\frac{|y - x^T\beta|}{s})``.
For simplicity we assume throughout this exercise that the intercept ``\beta_0 = 0``
and does not need to be fitted.

(a) Generate a training set of 100 points with the following data generator.
Notice that the noise follows a Laplace distribution instead of a normal distribution.
For once, we do not use a `DataFrame` here but represent the input explicitly as a matrix and the full dataset as a `NamedTuple`. If you run `data = data_generator()`, you can access the input matrix as `data.x` and the output vector as `data.y`.
```julia
function data_generator(; n = 100, β = [1., 2., 3.])
    x = randn(n, 3)
    y = x * β .+ rand(Laplace(0, 0.3), n)
    (x = x, y = y)
end
```

(b) Calculate with paper and pencil the negative log-likelihood loss. Apply transformations to the negative log-likelihood function to obtain a good loss function for gradient descent based on the practical considerations in the slides.

(c) Write the code to compute the loss on the training set for a given
parameter vector. *Hint:* use matrix multiplication, e.g. `data.x * β`.

(d) Perform gradient descent on the training set. Plot the learning curve to see
whether gradient descent has converged. If you see large fluctuations at the end
of training, decrease the learning rate. If the learning curve is not flat at
the end, increase the maximal number of steps. To see well the loss towards the end of gradient descent it is advisable to use log-scale for the y-axis (`yscale = :log10`).

(e) Estimate the coefficients with the standard linear regression.
Hint: do not forget that we fit without intercept (use `fit_intercept = false` in the `LinearRegressor`).

(f) Compare which method (d) or (e) found parameters closer to the one of our data generating process `[1, 2, 3]` and explain your finding.
"

# ╔═╡ 96e37eae-4b21-4f20-8a4e-091c0a95c92a
md"""
# Mathematical Optimization with JuMP.jl

The JuMP.jl package is an ambitious implementation of a modelling language for optimization problems in Julia.

In that sense, it is more like an AMPL (or Pyomo) built on top of the Julia language with macros, and able to use a variety of different commerical and open source solvers.

If you have a linear, quadratic, conic, mixed-integer linear, etc. problem then this will likely be the ideal “meta-package” for calling various solvers.

For nonlinear problems, the modelling language may make things difficult for complicated functions (as it is not designed to be used as a general-purpose nonlinear optimizer).

See the [quick start guide](http://www.juliaopt.org/JuMP.jl/0.18/quickstart.html) for more details on all of the options.

The following is an example of calling a linear objective with a nonlinear constraint (provided by an external function).
Here Ipopt stands for Interior Point OPTimizer, a [nonlinear solver](https://github.com/JuliaOpt/Ipopt.jl) in Julia
"""

# ╔═╡ 3945d2c8-ce3b-4013-8701-5ab7788d0248
begin
	# solve
	# max( x[1] + x[2] )
	# st sqrt(x[1]^2 + x[2]^2) <= 1
	
	function squareroot(x) # pretending we don't know sqrt()
	    z = x # Initial starting point for Newton’s method
	    while abs(z*z - x) > 1e-13
	        z = z - (z*z-x)/(2z)
	    end
	    return z
	end
	m = Model(Ipopt.Optimizer)
	# need to register user defined functions for AD
	JuMP.register(m,:squareroot, 1, squareroot, autodiff=true)
	
	@variable(m, x[1:2], start=0.5) # start is the initial condition
	@objective(m, Max, sum(x))
	@NLconstraint(m, squareroot(x[1]^2+x[2]^2) <= 1)
	@show JuMP.optimize!(m)
end

# ╔═╡ df7eae08-a1bf-4604-9dba-5e7e0fc933f1
md"""
As another example let us look at how we could use JuMP to solve the least squares problem stated earlier in this notebook.
"""

# ╔═╡ ea1713bb-fc34-4d79-833c-b288dbf0c3a3
let
	beta = [1.0, 2.2] # population parameters.

	# Generate noisy Data to perform Least Squares on.
	n = 100
	A = [ones(n) rand(n)]
	b = A*beta .+ randn(n)

	#min (Ax - b)'(Ax - b)
	m = Model(Ipopt.Optimizer)
	@variable(m, z[1:n])
	@variable(m, x[1:2])
	@constraint(m, z .== A * x - b)
	@objective(m, Min, z' * z)
	@show JuMP.optimize!(m)
	println("beta1 = ", JuMP.value(x[1]), " beta2 = ", JuMP.value(x[2]))
	
end


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
Ipopt = "b6b21f68-93f8-5de0-b562-5493be1d77c9"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MLJ = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
MLJLinearModels = "6ee0df7b-362f-4a72-a706-9e79364fb692"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
BenchmarkTools = "~1.3.1"
DataFrames = "~1.3.4"
Distributions = "~0.25.62"
FiniteDifferences = "~0.12.24"
Flux = "~0.13.3"
ForwardDiff = "~0.10.30"
Ipopt = "~1.0.2"
JuMP = "~1.1.0"
MLJ = "~0.18.2"
MLJLinearModels = "~0.6.3"
Plots = "~1.29.0"
PlutoUI = "~0.7.39"
ReverseDiff = "~1.14.0"
Zygote = "~0.6.40"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.ARFFFiles]]
deps = ["CategoricalArrays", "Dates", "Parsers", "Tables"]
git-tree-sha1 = "e8c8e0a2be6eb4f56b1672e46004463033daa409"
uuid = "da404889-ca92-49ff-9e8b-0aa6b4d38dc8"
version = "1.4.1"

[[deps.ASL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6252039f98492252f9e47c312c8ffda0e3b9e78d"
uuid = "ae81ac8f-d209-56e5-92de-9978fef736f9"
version = "0.1.3+0"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f1d9bc1c08f9f4a8fa92e3ea3cb50153a1b40d4"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.1.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Accessors]]
deps = ["Compat", "CompositionsBase", "ConstructionBase", "Future", "LinearAlgebra", "MacroTools", "Requires", "Test"]
git-tree-sha1 = "0264a938934447408c7f0be8985afec2a2237af4"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.11"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArrayInterface]]
deps = ["ArrayInterfaceCore", "Compat", "IfElse", "LinearAlgebra", "Static"]
git-tree-sha1 = "3b49db3cac86ed78cbd1653b6d610a30425b113c"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "6.0.7"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "d3a275e927d411e054c4192e5aca03998c233e94"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.7"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "a598ecb0d717092b5539dbbe890c98bac842b072"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.2.0"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "b15a6bc52594f5e4a3b825858d1089618871bf9d"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.36"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "4c10eee4af024676200bc7752e536f858c6b8f93"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.1"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "TimerOutputs"]
git-tree-sha1 = "925a16b909fdae16920c1319feadecffb6695b9d"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "3.10.1"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "5f5a975d996026a8dd877c35fe26a7b8179c02ba"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.6"

[[deps.CategoricalDistributions]]
deps = ["CategoricalArrays", "Distributions", "Missings", "OrderedCollections", "Random", "ScientificTypesBase", "UnicodePlots"]
git-tree-sha1 = "8c340dc71d2dc9177b1f701726d08d2255d2d811"
uuid = "af321ab8-2d2e-40a6-b165-3d674595d28e"
version = "0.1.5"

[[deps.ChainRules]]
deps = ["ChainRulesCore", "Compat", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics"]
git-tree-sha1 = "e9023f88b1655ffc6a4aaef2502878e8116151ef"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.35.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9489214b993cd42d17f44c36e359bf6a7c919abf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "2e62a725210ce3c3c2e1a3080190e7ca491f18d7"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.7.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "7297381ccb5df764549818d9a7d57e45f1057d30"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.18.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "0f4e115f6f34bbe43c19751c90a38b2f380637b9"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.3"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "87e84b2293559571802f97dd9c94cfd6be52c5e5"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.44.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "8ccaa8c655bc1b83d2da4d569c9b28254ababd6e"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.2"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "daa21eb85147f72e41f6352a57fccea377e310a9"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.4"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "28d605d9a0ac17118fe2c5e9ce0fbb76c3ceb120"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "0ec161f87bf4ab164ff96dfacf4be8ffff2375fd"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.62"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.EarlyStopping]]
deps = ["Dates", "Statistics"]
git-tree-sha1 = "98fdf08b707aaf69f524a6cd0a67858cefe0cfb6"
uuid = "792122b4-ca99-40de-a6bc-6742525f08b6"
version = "0.3.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "4391d3ed58db9dc5a9883b23a0578316b4798b1f"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.0"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "9267e5f50b0e12fdfd5a2455534345c4cf2c7f7a"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.14.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.FiniteDiff]]
deps = ["ArrayInterfaceCore", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "4fc79c0f63ddfdcdc623a8ce36623346a7ce9ae4"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.12.0"

[[deps.FiniteDifferences]]
deps = ["ChainRulesCore", "LinearAlgebra", "Printf", "Random", "Richardson", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "0ee1275eb003b6fc7325cb14301665d1072abda1"
uuid = "26cc04aa-876d-5657-8c51-4c34ba976000"
version = "0.12.24"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Flux]]
deps = ["Adapt", "ArrayInterface", "CUDA", "ChainRulesCore", "Functors", "LinearAlgebra", "MLUtils", "MacroTools", "NNlib", "NNlibCUDA", "Optimisers", "ProgressLogging", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "Test", "Zygote"]
git-tree-sha1 = "62350a872545e1369b1d8f11358a21681aa73929"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.13.3"

[[deps.FoldsThreads]]
deps = ["Accessors", "FunctionWrappers", "InitialValues", "SplittablesBase", "Transducers"]
git-tree-sha1 = "eb8e1989b9028f7e0985b4268dabe94682249025"
uuid = "9c68100b-dfe1-47cf-94c8-95104e173443"
version = "0.1.1"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "2f18915445b248731ec5db4e4a17e451020bf21e"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.30"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "cabd77ab6a6fdff49bfd24af2ebe76e6e018a2b4"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.0.0"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "b5c7fe9cea653443736d264b85466bad8c574f4a"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.9.9"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "241552bc2209f0fa068b6415b1942cc0aa486bcc"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.2"

[[deps.Functors]]
git-tree-sha1 = "223fffa49ca0ff9ce4f875be001ffe173b2b7de4"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.8"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GPUArrays]]
deps = ["Adapt", "LLVM", "LinearAlgebra", "Printf", "Random", "Serialization", "Statistics"]
git-tree-sha1 = "c783e8883028bf26fb05ed4022c450ef44edd875"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.3.2"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "d8c5999631e1dc18d767883f621639c838f8e632"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.15.2"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "b316fd18f5bc025fedcb708332aecb3e13b9b453"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.3"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1e5490a51b4e9d07e8b04836f6008f46b48aaa87"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.3+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "SpecialFunctions", "Test"]
git-tree-sha1 = "cb7099a0109939f16a4d3b572ba8396b1f6c7c31"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.10"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "af14a478780ca78d5eb9908b263023096c2b9d64"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.6"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "336cc738f03e069ef2cac55a104eb823455dca75"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.4"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.Ipopt]]
deps = ["Ipopt_jll", "MathOptInterface"]
git-tree-sha1 = "8b7b5fdbc71d8f88171865faa11d1c6669e96e32"
uuid = "b6b21f68-93f8-5de0-b562-5493be1d77c9"
version = "1.0.2"

[[deps.Ipopt_jll]]
deps = ["ASL_jll", "Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "MUMPS_seq_jll", "OpenBLAS32_jll", "Pkg"]
git-tree-sha1 = "e3e202237d93f18856b6ff1016166b0f172a49a8"
uuid = "9cc047cb-c261-5740-88fc-0cf96f7bdcc7"
version = "300.1400.400+0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IterationControl]]
deps = ["EarlyStopping", "InteractiveUtils"]
git-tree-sha1 = "d7df9a6fdd82a8cfdfe93a94fcce35515be634da"
uuid = "b3c1a2ee-3fec-4384-bf48-272ea71de57c"
version = "0.5.3"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1169632f425f79429f245113b775a0e3d121457c"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.JuMP]]
deps = ["Calculus", "DataStructures", "ForwardDiff", "LinearAlgebra", "MathOptInterface", "MutableArithmetics", "NaNMath", "OrderedCollections", "Printf", "SparseArrays", "SpecialFunctions"]
git-tree-sha1 = "67740bc79baa2104aa839f24181ea8110bb4f0f8"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.1.0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "10a20c556107dc5833d3bb7c5e45c4a6e191bd28"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.13.0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg", "TOML"]
git-tree-sha1 = "771bfe376249626d3ca12bcd58ba243d3f961576"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.16+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "46a39b9c58749eefb5f2dc1178cb8fab5332b1ab"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.15"

[[deps.LatinHypercubeSampling]]
deps = ["Random", "StableRNGs", "StatsBase", "Test"]
git-tree-sha1 = "42938ab65e9ed3c3029a8d2c58382ca75bdab243"
uuid = "a5e1c1ea-c99a-51d3-a14d-a9a37257b02d"
version = "1.8.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "f4d24f461dacac28dcd1f63ebd88a8d9d0799389"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "f27132e551e959b3667d8c93eae90973225032dd"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.1.1"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearMaps]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics"]
git-tree-sha1 = "9e9c3fa4de0b4f146d97eed3485711928789865b"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.6.2"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LossFunctions]]
deps = ["InteractiveUtils", "Markdown", "RecipesBase"]
git-tree-sha1 = "53cd63a12f06a43eef6f4aafb910ac755c122be7"
uuid = "30fc2ffe-d236-52d8-8643-a9d8f7c094a7"
version = "0.8.0"

[[deps.METIS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "1d31872bb9c5e7ec1f618e8c4a56c8b0d9bddc7e"
uuid = "d00139f3-1899-568f-a2f0-47f597d42d70"
version = "5.1.1+0"

[[deps.MLJ]]
deps = ["CategoricalArrays", "ComputationalResources", "Distributed", "Distributions", "LinearAlgebra", "MLJBase", "MLJEnsembles", "MLJIteration", "MLJModels", "MLJTuning", "OpenML", "Pkg", "ProgressMeter", "Random", "ScientificTypes", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "fba4aaf41e614fb03ce5aee8b08b0bb66b4e9b67"
uuid = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
version = "0.18.2"

[[deps.MLJBase]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Dates", "DelimitedFiles", "Distributed", "Distributions", "InteractiveUtils", "InvertedIndices", "LinearAlgebra", "LossFunctions", "MLJModelInterface", "Missings", "OrderedCollections", "Parameters", "PrettyTables", "ProgressMeter", "Random", "ScientificTypes", "Serialization", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "5907d51aa8a276a3161dfa9f3f301236efb14415"
uuid = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
version = "0.20.4"

[[deps.MLJEnsembles]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Distributed", "Distributions", "MLJBase", "MLJModelInterface", "ProgressMeter", "Random", "ScientificTypesBase", "StatsBase"]
git-tree-sha1 = "5b06d46c00da2eb0f2cc315a780fc3dcca28fcd5"
uuid = "50ed68f4-41fd-4504-931a-ed422449fee0"
version = "0.3.0"

[[deps.MLJIteration]]
deps = ["IterationControl", "MLJBase", "Random", "Serialization"]
git-tree-sha1 = "024d0bd22bf4a5b273f626e89d742a9db95285ef"
uuid = "614be32b-d00c-4edb-bd02-1eb411ab5e55"
version = "0.5.0"

[[deps.MLJLinearModels]]
deps = ["DocStringExtensions", "IterativeSolvers", "LinearAlgebra", "LinearMaps", "MLJModelInterface", "Optim", "Parameters"]
git-tree-sha1 = "2b1382afa4d7e711363e0960248e74baa9b86d4e"
uuid = "6ee0df7b-362f-4a72-a706-9e79364fb692"
version = "0.6.3"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "74d7fb54c306af241c5f9d4816b735cb4051e125"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.4.2"

[[deps.MLJModels]]
deps = ["CategoricalArrays", "CategoricalDistributions", "Dates", "Distances", "Distributions", "InteractiveUtils", "LinearAlgebra", "MLJModelInterface", "Markdown", "OrderedCollections", "Parameters", "Pkg", "PrettyPrinting", "REPL", "Random", "ScientificTypes", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "8291b42d6bf744dda0bfb16b6f0befbae232a1fa"
uuid = "d491faf4-2d78-11e9-2867-c94bc002c0b7"
version = "0.15.9"

[[deps.MLJTuning]]
deps = ["ComputationalResources", "Distributed", "Distributions", "LatinHypercubeSampling", "MLJBase", "ProgressMeter", "Random", "RecipesBase"]
git-tree-sha1 = "b61910708c28d783f214bbab33c887dcc6f7958a"
uuid = "03970b2e-30c4-11ea-3135-d1576263f10f"
version = "0.7.1"

[[deps.MLStyle]]
git-tree-sha1 = "2041c1fd6833b3720d363c3ea8140bffaf86d9c4"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.12"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "DelimitedFiles", "FLoops", "FoldsThreads", "Random", "ShowCases", "Statistics", "StatsBase"]
git-tree-sha1 = "95ab49a8c9afb6a8a0fc81df25617a6798c0fb73"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.2.5"

[[deps.MUMPS_seq_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "METIS_jll", "OpenBLAS32_jll", "Pkg"]
git-tree-sha1 = "29de2841fa5aefe615dea179fcde48bb87b58f57"
uuid = "d7ed1dd3-d0ae-5e8e-bfb4-87a502085b8d"
version = "5.4.1+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.MarchingCubes]]
deps = ["StaticArrays"]
git-tree-sha1 = "3bf4baa9df7d1367168ebf60ed02b0379ea91099"
uuid = "299715c1-40a9-479a-aaf9-4a633d36f717"
version = "0.1.3"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "49c71041d24803536113f69d7bfd1dac5375b06e"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.3.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "6bb7786e4f24d44b4e29df03c69add1b63d88f01"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "4050cd02756970414dab13b55d55ae1826b19008"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.0.2"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "50310f934e55e5ca3912fb941dec199b49ca9b68"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.2"

[[deps.NNlib]]
deps = ["Adapt", "ChainRulesCore", "Compat", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "f89de462a7bc3243f95834e75751d70b3a33e59d"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.5"

[[deps.NNlibCUDA]]
deps = ["CUDA", "LinearAlgebra", "NNlib", "Random", "Statistics"]
git-tree-sha1 = "e161b835c6aa9e2339c1e72c3d4e39891eac7a4f"
uuid = "a00861dc-f156-4864-bf3c-e6376f28a68d"
version = "0.2.3"

[[deps.NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS32_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c6c2ed4b7acd2137b878eb96c68e63b76199d0f"
uuid = "656ef2d0-ae68-5445-9ca0-591084a874a2"
version = "0.3.17+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenML]]
deps = ["ARFFFiles", "HTTP", "JSON", "Markdown", "Pkg"]
git-tree-sha1 = "06080992e86a93957bfe2e12d3181443cedf2400"
uuid = "8b6db2d4-7670-4922-a472-f9537c81ab66"
version = "0.2.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "7a28efc8e34d5df89fc87343318b0a8add2c4021"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "26f58049054343c8103d67a5530284a35f1186cb"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.2.5"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "3411935b2904d5ad3917dee58c03f0d9e6ca5355"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.11"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "1285416549ccfcdf0c50d4997a94331e88d68413"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "d457f881ea56bbfa18222642de51e0abf67b9027"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.29.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyPrinting]]
git-tree-sha1 = "4be53d093e9e37772cc89e1009e8f6ad10c4681b"
uuid = "54e16d92-306c-5ea0-a30b-337be88ac337"
version = "0.4.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "afeacaecf4ed1649555a19cb2cad3c141bbc9474"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.5.0"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.ReverseDiff]]
deps = ["ChainRulesCore", "DiffResults", "DiffRules", "ForwardDiff", "FunctionWrappers", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "Random", "SpecialFunctions", "StaticArrays", "Statistics"]
git-tree-sha1 = "bed55b9e6be9a7fd8012d9345774445605ff8ba3"
uuid = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
version = "1.14.0"

[[deps.Richardson]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e03ca566bec93f8a3aeb059c8ef102f268a38949"
uuid = "708f8203-808e-40c0-ba2d-98a6953ed40d"
version = "1.4.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.ScientificTypes]]
deps = ["CategoricalArrays", "ColorTypes", "Dates", "Distributions", "PrettyTables", "Reexport", "ScientificTypesBase", "StatisticalTraits", "Tables"]
git-tree-sha1 = "ba70c9a6e4c81cc3634e3e80bb8163ab5ef57eb8"
uuid = "321657f4-b219-11e9-178b-2701a2544e81"
version = "3.0.0"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "38d88503f695eb0301479bc9b0d4320b378bafe5"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "a9e798cae4867e3a41cae2dd9eb60c047f1212db"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.6"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[deps.StableRNGs]]
deps = ["Random", "Test"]
git-tree-sha1 = "3be7d49667040add7ee151fefaf1f8c04c8c8276"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.0"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "5d2c08cef80c7a3a8ba9ca023031a85c263012c5"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.6.6"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "cd56bf18ed715e8b09f06ef8c6b781e6cdc49911"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.4"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "271a7fea12d319f23d55b785c51f6876aadb9ac0"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.0.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c82aaa13b44ea00134f8c9c89819477bd3986ecd"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.3.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "9abba8f8fb8458e9adf07c8a2377a070674a24f1"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.8"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "7638550aaea1c9a1e86817a231ef0faa9aca79bd"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.19"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "c76399a3bbe6f5a88faa33c8f8a65aa631d95013"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.73"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.UnicodePlots]]
deps = ["ColorTypes", "Contour", "Crayons", "Dates", "FileIO", "FreeTypeAbstraction", "LazyModules", "LinearAlgebra", "MarchingCubes", "NaNMath", "Printf", "SparseArrays", "StaticArrays", "StatsBase", "Unitful"]
git-tree-sha1 = "f409b707e8f901cf20b2d3eab5ee393c2f43f2de"
uuid = "b8865327-cd53-5732-bb35-84acbb429228"
version = "2.12.1"

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "b649200e887a487468b71821e2644382699f1b0f"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.11.0"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "IRTools", "InteractiveUtils", "LinearAlgebra", "MacroTools", "NaNMath", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "a49267a2e5f113c7afe93843deea7461c0f6b206"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.40"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╠═e03882f9-843e-4552-90b1-c47b6cbba19b
# ╠═7aa547f8-25d4-488d-9fc3-f633f7f03f57
# ╟─28408f30-dfa4-4272-a215-16812eaa4398
# ╟─9f18f610-79e2-403a-a792-fe2dafd2054a
# ╠═2803f283-4800-47d5-8822-f7a45d945c3e
# ╠═7cf19059-f00d-4e04-a7fe-41274f3c0d8a
# ╠═7630772c-c505-4ea3-a7a8-9cf54aee189a
# ╠═9d04de9a-6220-4a16-be52-7714f21f193b
# ╟─a976b898-37ba-430a-b8b1-2b6c55dfd0c6
# ╠═3ea9881c-91c1-4aac-a2ab-4097e52d9ddf
# ╠═8a660a1f-b84b-4ef9-b5fb-2f1740b354d3
# ╠═196c7318-3b97-4ea4-a80d-17d2e9567193
# ╟─bcc646a2-287d-48db-95f5-1ebf6f0d5f4c
# ╠═2a0bdd82-2de8-4990-9357-52c8270e592d
# ╟─b5f0e109-ac66-40f7-9962-aa89b8b5de65
# ╠═b4039c01-ddee-4818-ae42-802eee4fa15a
# ╟─4a53fc66-6822-4f85-81a7-319551e4882b
# ╠═b08dffa2-686c-4651-8e06-ee2362ead7c3
# ╠═219399db-4ab5-43fe-bab4-a24fc2608f72
# ╟─9c2c5079-9357-4765-bf61-2db9b096e667
# ╠═1137bb34-e4d2-4344-9f85-8792158e505c
# ╟─96a5358f-c0c5-4c87-893d-812075a6f983
# ╠═fc1d9f67-140d-4484-af75-d6a7996ccb2d
# ╟─b2ae0ab8-42df-4e1f-890a-b321fd8d71ea
# ╠═35e51599-b306-43b1-9139-0431a70ec88c
# ╟─70dabe04-94cc-4900-a893-2131fba389ed
# ╠═119d8b81-96a2-4e62-aac3-b7df5bf77ee2
# ╟─2022c744-a0ab-4817-a45b-7fe94360451c
# ╠═0ae02b75-435a-4ef3-8fdf-5422653c84cd
# ╠═33c8744d-1205-46e2-93a7-85ef35efb081
# ╠═d420f077-d0a5-4361-b23f-bfb5a8ff450e
# ╠═4274811c-a361-440a-9f8d-1bdb607ebbf5
# ╟─7706d51c-7244-4077-96b3-4432a8093e59
# ╠═a723876e-ae49-4b3d-8cd6-d93efb774ef5
# ╠═8764c694-5fd6-49ea-aa08-8792f00e7593
# ╠═f595bf3e-c063-4d29-9123-ad62a8a77e97
# ╟─791b26ad-f259-4cd0-afc1-b5c658ad789d
# ╠═16cca6d5-d933-4203-b8a5-6049ad00a3b0
# ╠═96ff3a8b-2aa9-476c-b3d9-6dc1cabbac36
# ╠═c4c855d0-745a-4519-8448-38e5993df5df
# ╠═d9b0e98f-d12a-4208-a902-f9c31f93ce91
# ╠═c780c4c5-50e3-4823-883b-87e2d33b34b5
# ╠═250a0377-cd24-4756-8b2e-da1f432c86b4
# ╠═adbcd6cb-3929-4ee0-ba74-e452e753572d
# ╠═677327cf-628f-446c-be22-b3ec8c89a785
# ╠═0e73bf21-b241-4589-a532-f8a182b826ff
# ╟─134b91d4-a26c-430a-bc3f-0cb41feb531e
# ╠═526192ca-5ed7-4e4b-b05c-53d706bcc032
# ╟─d2d46434-b6e8-4685-8960-0024db9c8539
# ╠═fcbac567-36db-40c5-9436-6315c0caa40a
# ╠═49e744dc-33b7-455a-9cf7-4e8e12f11152
# ╠═dcc76840-6552-4a86-8268-1291ab8ea0d3
# ╟─3ca1e8f3-d69f-454f-b917-4bbe2dcfce01
# ╠═b1fc14bb-1fd2-4739-a761-7a605fd4559b
# ╟─bbc0b514-4789-44d1-8d90-9fc325d9ad6b
# ╟─eb9e2b3f-dc4b-49a2-a611-b45f2918adcf
# ╠═402bb576-8945-403e-a9a6-fd5bfb8016bc
# ╟─cd5f079f-a06d-4d55-9666-e2b05ddf8989
# ╠═93699313-5e42-48a7-abc6-ad146e5bdcdd
# ╟─33bdc912-e46a-4310-9184-733be7871768
# ╟─979feb42-c328-44b5-b519-8e8cda474140
# ╠═c448238c-f712-4af3-aedb-cec3a2c1a73e
# ╠═6e38a3f6-a592-4dc1-a6c4-d0f0050c9399
# ╠═9a61332f-cdc2-4129-b7b5-5ab54ba387a3
# ╟─4b3ac87f-39f6-4e6c-b7b3-536925e9b112
# ╟─01221937-b6d9-4dac-be90-1b8a1c5e9d87
# ╟─f5e27275-8751-4e80-9888-c3d22d8e80e3
# ╟─ca1301fd-8978-44d6-bfae-e912f939d7a8
# ╠═e821fb15-0bd3-4fa7-93ea-692bf05097b5
# ╟─75528011-05d9-47dc-a37b-e6bb6be52c25
# ╟─7541c203-f0dc-4445-9d2a-4cf16b7e912a
# ╟─913cf5ee-ca1e-4063-bd34-6cccd0cc548b
# ╠═eb289254-7167-4183-a4d0-52f68be66b04
# ╟─166472c5-c0f4-4261-a476-4c9b0f82abd6
# ╟─08a9418f-786e-4992-b1a5-04cf9060f8fe
# ╠═dc57d700-2a82-4ab0-9bd2-6ce622cb0fa5
# ╠═c72dc506-fb1d-42ee-83cf-cc49753ecd4f
# ╟─075f35cf-4271-4676-b9f3-e2bcf610c2d1
# ╟─0d431c00-9eef-4ce4-9542-9571728d1501
# ╠═e0cc188c-9c8f-47f1-b1fe-afc2a578973d
# ╟─96e37eae-4b21-4f20-8a4e-091c0a95c92a
# ╠═de2ea96e-df6f-4799-b912-07de7e5b965a
# ╠═3945d2c8-ce3b-4013-8701-5ab7788d0248
# ╠═df7eae08-a1bf-4604-9dba-5e7e0fc933f1
# ╠═ea1713bb-fc34-4d79-833c-b288dbf0c3a3
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
