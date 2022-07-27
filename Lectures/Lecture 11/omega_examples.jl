### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ df9ef4ac-00f0-11ed-0be8-37f9d537e4dd
begin
	using Pkg
	Pkg.develop(["Omega", "Distributions"])
	Pkg.add(["PrettyPrinting", "GraphPlot", "Graphs", "Plots", "Images"])
	Pkg.add(["CSV", "DataFrames"])
end

# ╔═╡ c72e14e9-9fad-4e71-8859-e8583762b9ff
using Omega, Distributions

# ╔═╡ b12f2f43-59a0-4ecf-8972-a09323db55d6
using DataFrames, CSV

# ╔═╡ ce37be62-874a-442b-8e18-84d5e3737d2d
using PrettyPrinting, GraphPlot, Graphs, Plots, Images

# ╔═╡ e8244915-859d-4b4a-a1e6-506c5951fb3c
md"""
## Kidney Stones Example
The underlying SCM is as follows - 
- `Z` is the size of the stone,
- `T` is the treatment
- `R` is the recovery
"""

# ╔═╡ fede2ee9-8693-4a71-9bd6-73653d8944d7
md"""
Table below shows a famous data set from kidney stone recovery [Charig et al., 1986]. Out of 700 patients, one half was treated with open surgery (treatment T = a, 78% recovery rate) and the other half with percutaneous nephrolithotomy (T = b, 83% recovery rate), a surgical procedure to remove kidney stones by a small puncture wound. If we do not know anything else than the overall recovery rates, and neglect side effects, for example, many people would prefer treatment b if they had to decide.
"""

# ╔═╡ e15f9463-ab62-443e-979a-660b4ef05985
load("table.png")

# ╔═╡ ce923a7c-d655-480c-ac4c-082dfbd72b4b
md"""
This is a classic example of Simpson’s paradox. The table reports the success rates of
two treatments for kidney stones [Bottou et al., 2013, Charig et al., 1986, tables I and II]. Although the overall success rate of treatment `b` seems better (any bold number is largest in its column), treatment `b` performs worse than treatment `a` on both patients with small kidney stones and patients with large kidney stones.
"""

# ╔═╡ 31ee2a19-886d-4ce6-921c-f625b648325d
md"""
The example is taken from the book - Elements of Causal Inference. 
"""

# ╔═╡ f7541377-d82a-4ab4-950d-f420a2528f77
md"""
Encoding the above information as SCM in Omega -
"""

# ╔═╡ bc027ed4-dcd4-46ea-9bb0-ee5f380997d2
Z = ifelse.((@~ Bernoulli(357/700)), :small, :large)

# ╔═╡ 6cdfc663-98e2-4cf0-b457-b35bcbe82127
begin
	function T_(ω::Ω)
		treatments = [:a, :b]
		if Z(ω) == :small
			return treatments[(@~ Bernoulli(87/(270+87)))(ω) + 1]
		else
			return treatments[(@~ Bernoulli(263/(263+80)))(ω) + 1]
		end
	end
	T = Variable(T_)
end

# ╔═╡ 80d563ae-079f-4994-82ef-98c0f0c33569
begin
	function R_(ω::Ω)
		if T(ω) == :a && Z(ω) == :small
			return (@~ Bernoulli(81/87))(ω)
		elseif T(ω) == :a && Z(ω) == :large  
			return (@~ Bernoulli(192/263))(ω)
		elseif T(ω) == :b && Z(ω) == :small
			return (@~ Bernoulli(234/270))(ω)
		elseif T(ω) == :b && Z(ω) == :large
			return (@~ Bernoulli(55/80))(ω)
		end
	end
	R = Variable(R_)
end

# ╔═╡ ea8a7597-b5fd-4003-a073-919da7c99b90
md"""
Since `T` and `R` are elaborate functions, it is recommended and often necessary to wrap them with `Variable` struct.
"""

# ╔═╡ 3abf8a3e-7977-4d4f-ad55-5a0e32ea23a0
md"""
We define a helper function `prob`, that takes the random variable and number of samples as inputs, to estimate the emperical probability (with `10000` as default number of samples).
"""

# ╔═╡ 2261c278-efb3-4649-b12a-1ecdca619046
prob(rv, nsamples = 10000) = mean(randsample(rv, nsamples))

# ╔═╡ b68c1f70-a450-4485-b224-37f6d46b26d6
md"""
Probability of recovery being `1` after intervening on treatment to be `a` is -
"""

# ╔═╡ cfbf5c30-2ef7-4aa6-8bae-8db37b2c35e4
P_a = prob((R |ᵈ (T => :a)) .== 1) # P(R = 1, do(T := a))

# ╔═╡ c1e57104-810f-48b9-bb51-5c289242791e
md"""
Similarly setting treatment to be `b` -
"""

# ╔═╡ c373afb2-d9ae-441a-9c44-c41cc79b72dc
P_b = prob((R |ᵈ (T => :b)) .== 1) # P(R = 1, do(T := b))

# ╔═╡ da058ae7-980e-46f7-82ee-e52eebf3f6b6
md"""
From the above two, we conclude can that we would rather go for treatment `a`.
Notice how the probabilities after intervention and conditioning are different:
"""

# ╔═╡ 1d8f2088-bf78-4c4a-9edc-8188e71b1960
prob(R |ᶜ (T .== :a)) # different from the value obtained after intervening on treatment to be a

# ╔═╡ 2ac288d0-ce20-4f1c-8d09-adc6efb11b52
prob(R |ᶜ (T .== :b)) # different from the value obtained after intervening on treatment to be b

# ╔═╡ 231668ea-045c-418b-baa3-80043e57d847
ACE = P_a - P_b

# ╔═╡ f670a623-aef3-4be6-ba24-8b534402f274
md"""
The quantity above is called average causal effect for binary treatments. It shows the effect of treatment on recovery as the difference between the two intervened probabilities. 

Formally, _Average Causal Effect_ (ACE) is defined as the average difference between potential outcomes under different treatments.

"""

# ╔═╡ 24cda179-980f-48bc-b3d8-281220c81d47
md"""
From the above model we could quite easily derive the causal effect, but this would not be the case if the SCM of the model is not available, which is often the case.

We often have data from the underlying SCM, from this data we could use various techniques to obtain the causal graph, but we can not accurately obtain the SCM. In such a situation, we could estimate the average causal effect using only the data and the causal graph if we can extract the valid adjustment set.
"""

# ╔═╡ 678660f3-d93d-4537-9b12-44dddae2cb30
md"""
We first create a toy dataset represented by the sbove SCM -
"""

# ╔═╡ a7d48618-bdf4-43dd-9268-6381e283ec23
data_model = @joint Z T R

# ╔═╡ 49e0cde3-a501-43d1-a7b9-dc7e2f12effc
n = 10000

# ╔═╡ 0c86641a-aa4d-4b79-8a73-24f6f2244da1
begin
	samples = randsample(data_model, n)
	Z_samples = map(x -> x.Z, samples)
	T_samples = map(x -> x.T, samples)
	R_samples = map(x -> x.R, samples)
end

# ╔═╡ 3952693a-9a77-4798-a29f-fa42bed8dd2f
md"""
Now that we have the dataset, we find the  average causal effect by computing the interventional probability by adjusting for confounding variables (here: `Z`: since it is a common cause for both `T` and `R`).
"""

# ╔═╡ ea8c3df9-28aa-43f4-8285-df2b5c0a56bd
md"""
$$P(y | do(Z = z)) = Σₓ P(y | z, x) P(x)$$
"""

# ╔═╡ 603e0351-4652-431f-8922-62028496e3f0
p_r_conditioned_on_t_z(t, z) = 
	mean([R_samples[i] == 1 for i in 1:n if (T_samples[i] == t && Z_samples[i] == z)])

# ╔═╡ ce98a74f-a5b3-43e4-975d-a608a6f5832a
p_z(i) = mean(Z_samples .== i)

# ╔═╡ 9cf11400-0556-4ac5-9b8e-d2a8300610a5
P(t) = 
	sum(Float64[p_r_conditioned_on_t_z(t, i) * p_z(i) for i in unique(Z_samples)])

# ╔═╡ 135afcda-d28a-4e40-ba14-b6a539fdd574
ACE_ = P(:a) - P(:b)

# ╔═╡ 88c13988-407c-4933-9849-5594d58c6446
isapprox(ACE, ACE_, atol = 0.9)

# ╔═╡ 4738f7c3-509f-4607-8130-5252e51d958b
md"""
## Law School Example
"""

# ╔═╡ f768924f-1549-44ad-9b11-704211f307ea
md"""
Law school dataset is a synthetic dataset that is used primarily in fairness-related work. We know the causal graph of the data, but not the underlying joint distributions.
"""

# ╔═╡ a622c27f-a230-42e2-9ac6-12096e7931c4
CSV.read("law_school.csv", DataFrame)

# ╔═╡ 81553151-d0a1-45c5-b1be-070651031fdc
begin
	g = DiGraph(5)
	add_edge!(g, 1 => 3)
	add_edge!(g, 1 => 4)
	add_edge!(g, 1 => 5)
	add_edge!(g, 2 => 3)
	add_edge!(g, 2 => 4)
	add_edge!(g, 2 => 5)
	add_edge!(g, 3 => 5)
	add_edge!(g, 4 => 5)
	gplot(g, nodelabel = [:sex, :race, :GPA, :LSAT, :FYA], layout = shell_layout)
end

# ╔═╡ a6418ea9-5c7e-4732-aa01-11f8751a21d6
md"""
As we see from the graph, to estimate the causal effect of `gpa` on `fya`, we need to adjust for the variables `sex` and `race` -
"""

# ╔═╡ 9722ab50-23a1-42b4-ba18-9ea3f67e0af4


# ╔═╡ Cell order:
# ╠═df9ef4ac-00f0-11ed-0be8-37f9d537e4dd
# ╠═c72e14e9-9fad-4e71-8859-e8583762b9ff
# ╠═b12f2f43-59a0-4ecf-8972-a09323db55d6
# ╠═ce37be62-874a-442b-8e18-84d5e3737d2d
# ╟─e8244915-859d-4b4a-a1e6-506c5951fb3c
# ╟─fede2ee9-8693-4a71-9bd6-73653d8944d7
# ╟─e15f9463-ab62-443e-979a-660b4ef05985
# ╟─ce923a7c-d655-480c-ac4c-082dfbd72b4b
# ╟─31ee2a19-886d-4ce6-921c-f625b648325d
# ╟─f7541377-d82a-4ab4-950d-f420a2528f77
# ╠═bc027ed4-dcd4-46ea-9bb0-ee5f380997d2
# ╠═6cdfc663-98e2-4cf0-b457-b35bcbe82127
# ╠═80d563ae-079f-4994-82ef-98c0f0c33569
# ╟─ea8a7597-b5fd-4003-a073-919da7c99b90
# ╟─3abf8a3e-7977-4d4f-ad55-5a0e32ea23a0
# ╠═2261c278-efb3-4649-b12a-1ecdca619046
# ╟─b68c1f70-a450-4485-b224-37f6d46b26d6
# ╠═cfbf5c30-2ef7-4aa6-8bae-8db37b2c35e4
# ╟─c1e57104-810f-48b9-bb51-5c289242791e
# ╠═c373afb2-d9ae-441a-9c44-c41cc79b72dc
# ╟─da058ae7-980e-46f7-82ee-e52eebf3f6b6
# ╠═1d8f2088-bf78-4c4a-9edc-8188e71b1960
# ╠═2ac288d0-ce20-4f1c-8d09-adc6efb11b52
# ╠═231668ea-045c-418b-baa3-80043e57d847
# ╟─f670a623-aef3-4be6-ba24-8b534402f274
# ╟─24cda179-980f-48bc-b3d8-281220c81d47
# ╟─678660f3-d93d-4537-9b12-44dddae2cb30
# ╠═a7d48618-bdf4-43dd-9268-6381e283ec23
# ╠═49e0cde3-a501-43d1-a7b9-dc7e2f12effc
# ╠═0c86641a-aa4d-4b79-8a73-24f6f2244da1
# ╟─3952693a-9a77-4798-a29f-fa42bed8dd2f
# ╟─ea8c3df9-28aa-43f4-8285-df2b5c0a56bd
# ╠═603e0351-4652-431f-8922-62028496e3f0
# ╠═ce98a74f-a5b3-43e4-975d-a608a6f5832a
# ╠═9cf11400-0556-4ac5-9b8e-d2a8300610a5
# ╠═135afcda-d28a-4e40-ba14-b6a539fdd574
# ╠═88c13988-407c-4933-9849-5594d58c6446
# ╟─4738f7c3-509f-4607-8130-5252e51d958b
# ╟─f768924f-1549-44ad-9b11-704211f307ea
# ╠═a622c27f-a230-42e2-9ac6-12096e7931c4
# ╟─81553151-d0a1-45c5-b1be-070651031fdc
# ╟─a6418ea9-5c7e-4732-aa01-11f8751a21d6
# ╠═9722ab50-23a1-42b4-ba18-9ea3f67e0af4
