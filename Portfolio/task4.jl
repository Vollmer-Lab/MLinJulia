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

# ╔═╡ 30f63647-779f-4c3b-877c-4e9b62fe7bcd
begin
	using Pkg
	Pkg.develop(url="https://github.com/zenna/Omega.jl#lang")
	using Omega
end

# ╔═╡ 770468bd-74bc-4c72-a4ae-239d966592a2
using PlutoUI, Distributions

# ╔═╡ 68201db3-08fd-4ce6-b7ef-cbcbab0b2702
using Graphs, GraphMakie, Makie.Colors, GLMakie

# ╔═╡ af08744d-3146-4eb0-89d4-6ea74ab7f61c
md"### Importing the required packages:"

# ╔═╡ 9f3da279-c520-4ac2-84ab-9cf49c19b3aa
md"Since Omega is still being developed, it must be used with `develop` as follows:"

# ╔═╡ cbbba552-9494-4ed5-8a6d-6a5a15513e0c
md"Importing the graphing packages:"

# ╔═╡ 100f9b53-13c6-449c-afc1-40d00fcef638
md"### Taking the SCM as a user input:"

# ╔═╡ a8872eeb-0de2-45ac-8298-06759414a894
timeofday = @~ UniformDraw([:morning, :afternoon, :evening])

# ╔═╡ a7850758-4457-4e65-93f8-fb8661a4810e
is_window_open = @~ Bernoulli()

# ╔═╡ bfa83267-3b2e-42b8-809e-a74a8e20b170
is_ac_on = @~ Bernoulli(0.3)

# ╔═╡ abcbbb2a-9484-460a-a25d-e2fe1c4ecef1
function inside_temp(ω::Ω)
  if is_ac_on(ω)
    (@~Normal(20.0, 1.0))(ω)
  else
    (@~Normal(25.0, 1.0))(ω)
  end
end

# ╔═╡ 88a02c9d-8838-4068-ba92-6ac28c35532e
function outside_temp(ω::Ω)
  if timeofday(ω) == :morning
    (@~ Normal(20.0, 1.0))(ω)
  elseif timeofday(ω) == :afternoon
    (@~ Normal(32.0, 1.0))(ω)
  else
    (@~ Normal(10.0, 1.0))(ω)
  end
end

# ╔═╡ 55f7c235-8b9c-4e11-ac94-b162623a4164
function thermostat(ω::Ω)
  if is_window_open(ω)
    (outside_temp(ω) + inside_temp(ω)) / 2.0
  else
    inside_temp(ω)
  end
end

# ╔═╡ 6143aec1-acd7-4d66-8208-c7fbef8d5ba4
md"You may now use the textbox feature of Pluto notebooks and `Meta.parse` - `eval` to use variables created previously within the textbox - "

# ╔═╡ 53df43b2-7f0d-49fc-bbbb-aebc6878eeaf
@bind rv html"<input type=text >"

# ╔═╡ c1c47954-3f56-43a2-95db-056fd1302ca3
rv′  = Meta.parse(rv)

# ╔═╡ cc5bedd0-1bb0-42a9-a50f-d5ed25ba0bb6
t = eval(rv′)

# ╔═╡ 3bd6856a-0205-4db0-ae1d-640ce9ed2f26
md"We can intervene on the variable being `true` (in this example taking the user-input as a Boolean random variable) - "

# ╔═╡ 8295b02c-46f3-48f9-81be-fd2e1ce02f27
mean(randsample(thermostat |ᵈ (t => true), 100)) # The value entered in rv is a random variable, ex: is_ac_on

# ╔═╡ 1a5d8ccc-2e40-483f-8c65-53694d2e67ed
md"### Graphing the SCM:"

# ╔═╡ c6c65f75-2f8d-4a85-9468-c7c36902c5be
dag = DiGraph(6)

# ╔═╡ bc34d266-3d39-4a65-8010-df95186ca99e
add_edge!(dag, 3, 4)

# ╔═╡ 94f37072-e2b5-4f1a-907d-279c7377e9d5
add_edge!(dag, 1, 5)

# ╔═╡ e1f6dc7c-3325-4dc1-b4bc-f67c9186c821
add_edge!(dag, 2, 6)

# ╔═╡ 4f79b368-21af-4a77-bc6e-7fc4494efd59
add_edge!(dag, 4, 6)

# ╔═╡ a016cfd9-ebec-4216-9f97-51019e89cdf0
add_edge!(dag, 5, 6)

# ╔═╡ 43e023fb-d9d5-4cd0-9efe-735c7bfc062c
nlabels = ["timeofday", "is_window_open", "is_ac_in", "inside_temp", "outside_temp", "thermostat"]

# ╔═╡ 51fdde5e-6b4f-4c9b-86cb-85eaf8ba5f70
graphplot(dag, nlabels=nlabels)

# ╔═╡ c9827e51-7ec1-43e3-9dc1-3113b2cbd26f
md"By the end of the task, the above plot must be interactive."

# ╔═╡ Cell order:
# ╟─af08744d-3146-4eb0-89d4-6ea74ab7f61c
# ╠═770468bd-74bc-4c72-a4ae-239d966592a2
# ╟─9f3da279-c520-4ac2-84ab-9cf49c19b3aa
# ╠═30f63647-779f-4c3b-877c-4e9b62fe7bcd
# ╟─cbbba552-9494-4ed5-8a6d-6a5a15513e0c
# ╠═68201db3-08fd-4ce6-b7ef-cbcbab0b2702
# ╟─100f9b53-13c6-449c-afc1-40d00fcef638
# ╠═a8872eeb-0de2-45ac-8298-06759414a894
# ╠═a7850758-4457-4e65-93f8-fb8661a4810e
# ╠═bfa83267-3b2e-42b8-809e-a74a8e20b170
# ╠═abcbbb2a-9484-460a-a25d-e2fe1c4ecef1
# ╠═88a02c9d-8838-4068-ba92-6ac28c35532e
# ╠═55f7c235-8b9c-4e11-ac94-b162623a4164
# ╟─6143aec1-acd7-4d66-8208-c7fbef8d5ba4
# ╠═53df43b2-7f0d-49fc-bbbb-aebc6878eeaf
# ╠═c1c47954-3f56-43a2-95db-056fd1302ca3
# ╠═cc5bedd0-1bb0-42a9-a50f-d5ed25ba0bb6
# ╟─3bd6856a-0205-4db0-ae1d-640ce9ed2f26
# ╠═8295b02c-46f3-48f9-81be-fd2e1ce02f27
# ╟─1a5d8ccc-2e40-483f-8c65-53694d2e67ed
# ╠═c6c65f75-2f8d-4a85-9468-c7c36902c5be
# ╠═bc34d266-3d39-4a65-8010-df95186ca99e
# ╠═94f37072-e2b5-4f1a-907d-279c7377e9d5
# ╠═e1f6dc7c-3325-4dc1-b4bc-f67c9186c821
# ╠═4f79b368-21af-4a77-bc6e-7fc4494efd59
# ╠═a016cfd9-ebec-4216-9f97-51019e89cdf0
# ╠═43e023fb-d9d5-4cd0-9efe-735c7bfc062c
# ╠═51fdde5e-6b4f-4c9b-86cb-85eaf8ba5f70
# ╟─c9827e51-7ec1-43e3-9dc1-3113b2cbd26f
