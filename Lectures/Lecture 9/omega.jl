### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ e0b80775-ca75-4d05-9863-a534c27f47ff
begin
	using Pkg
	Pkg.develop("Omega")
	Pkg.add("UnicodePlots")
	Pkg.add("Distributions")
	Pkg.add("FreqTables")
	using Omega, Distributions, UnicodePlots, FreqTables
end

# ╔═╡ f8041cae-ef14-11ec-3bf2-67b19132ea93
md"""
Omega is a probabilistic programming language (PPL). It is a formal language that is designed to express the kinds of knowledge individuals have about the world. This is done by the means of generative models. 

The generative approach to cognition posits that some mental representations are more like theories in this way: they capture general descriptions of how the world works. A generative model describes a process, usually, one by which observable data is generated, that represents knowledge about the causal structure of the world. These generative processes are simplified “working models” of a domain. Other mental computations operate over these generative models to draw inferences: many different questions can be answered by interrogating the mental model.
"""

# ╔═╡ d85d114a-be83-4ee6-a4c3-136b34c82112
md"# Building Generative Models"

# ╔═╡ fda57285-7d1d-4618-a0e4-8f1946936115
md"""
The key idea is that in Omega you can express stochastic processes (random variables) and not just deterministic functions (like logical AND gate). Deterministic models return unique values for the same set of inputs whereas stochastic models have random variables as inputs, which result in random outputs. Most models we encounter are better described by random variables, for example, tossing a coin. We can simulate a coin toss in Omega, unlike in a deterministic programming language.
"""

# ╔═╡ 528c88c9-a731-4628-b01b-7b6807847bfd
md"""
A basic object in Omega is a class representing random variables, from which you can generate random variables. For instance, a random variable that is of `Bernoulli` distribution is generated as given below -
"""

# ╔═╡ 0f085170-058a-45df-b377-1616c52c7875
a = @~ Bernoulli()

# ╔═╡ b08ac749-f261-4b3a-b269-759351025bd5
md"Here, `Bernoulli()` is a random class and to get a member (a random variable, which we can sample from) of the class we use `@~`. The long-form of writing this expression is -"

# ╔═╡ 6b2c770c-1dac-4a06-bdbd-585a194203e3
id = 0

# ╔═╡ bdd7cb5f-1e11-4886-ad82-122e2ae598a7
a_ = id ~ Bernoulli() 

# ╔═╡ 2e89bd6f-1ccf-4440-88fa-157d06121713
md"where `0` is an ID, which is used to refer to the particular random variable in Omega. IDs may be tuples, numbers, strings or julia symbols. `@~` is a syntactic sugar that automatically selects an ID."

# ╔═╡ 713af13a-e328-4719-9be2-b1baf181ae25
md"### Sampling "

# ╔═╡ 4fcea488-67d0-4a7f-8411-56f0db1c5713
md"`randsample` is used to **sample** from a random variable:"

# ╔═╡ 989c9194-ed60-4525-a371-f5d63c8dfa89
randsample(a)

# ╔═╡ 2e17790c-651a-4546-88e8-00dfd9bcfacc
md"Run this program a few times. You will get back a different sample on each execution."

# ╔═╡ 1a31c200-569f-46d1-8cd2-ee3ce5008ac1
md"""
Random variables in Omega are functions of the form - `f(ω::Ω)` and `randsample` can used to sample from them.
"""

# ╔═╡ 508eaa3b-7146-437c-9efe-f493f28bd81b
md"### Primitive and Composite Classes in Omega"

# ╔═╡ 7b111711-cbb5-40c8-89bd-e54f3855df54
md"""##### Primitive Classes
Omega comes with a set of built-in primitive random variable classes, such as `StdNormal` and `StdUniform`.  There are parameterless infinite sets of random variables.
"""

# ╔═╡ ba91848f-1abb-4728-ba97-dcdc380a7e69
As = StdUniform{Float64}()

# ╔═╡ 0091feef-8c36-4d72-aa93-3a97c7265dc4
rv = 1 ~ As

# ╔═╡ 557d9b90-769e-4ab0-8f22-172c7ce83cdf
randsample(rv)

# ╔═╡ 5e41d415-349e-416c-b30a-80ff38b60011
md"""
##### Composite Class

A class in Omega is actually just a function of the form `f(id, ω)`. Of course, you can specify your own classes simply by constructing a function.
"""

# ╔═╡ 4eae034d-2205-4b7d-acae-92d99868bf19
μ = 2 ~ StdNormal{Float64}()

# ╔═╡ abec0967-0b47-49e5-9678-aa64959f12d9
x(id, ω::Ω) = (id ~ Normal(μ(ω), 1))(ω)

# ╔═╡ f7274e22-ab59-4fdb-96a9-73873e1ef283
x_ = 3 ~ x

# ╔═╡ 580d51dd-5eb0-48b1-9827-aa036328b925
md"`x_` is a random variable of the class `x`"

# ╔═╡ 075533af-bff9-453c-91ab-e5c24c8aed9d
randsample((μ, x_))

# ╔═╡ 69bde3f4-d211-4185-ade2-ffc44893aa56
md"Alternatively, `@joint` is used to create the joint distribution of variables:"

# ╔═╡ f526ddd6-ae51-4ff0-8c07-0098c2253d66
joint = @joint μ  _

# ╔═╡ fad059c0-484c-4abb-bc82-e46442ec98c1
randsample(joint)

# ╔═╡ 2dd0abb4-6692-4b89-810d-aeed7cbd3e05
md"""
Every time we create a new (independent) random variable, we need to provide a unique id to it. We could instead use `@~` to generate unique `id`s automatically.
"""

# ╔═╡ 41f297fd-3b53-412d-8ab4-8560c44c1229
md"Below we describe a process that samples a number adding up several independent Bernoulli distributions:"

# ╔═╡ 4f2bef6a-51de-4f44-9ba1-ad3fbee0f722
b_sum = (@~ Bernoulli()) .+ (@~ Bernoulli()) .+ (@~ Bernoulli())

# ╔═╡ 2cab677e-ff84-4d0e-a880-7b8fce90b147
md"We use `.+` or `pw` to operate on random variables."

# ╔═╡ 0c8164ae-20e2-4590-8c9a-2e0f25d28505
randsample(b_sum)

# ╔═╡ 5d8e4aba-1a30-4212-8d80-10036faacdf7
md"We have constructed a random variable that is a sum of three random variables and sampled from it. We can construct such complex random variables from the primitive ones."

# ╔═╡ 11d7f790-b50a-4d89-bd8d-b08db06b5056
histogram(randsample(b_sum, 1000))

# ╔═╡ 86a7ce00-c37a-431b-8121-e5464fb0d4b3
md"Complex functions can also have other arguments. Here is a random variable that will only sometimes double its input:"

# ╔═╡ 48c9cbfe-07f9-4b6f-903e-03888a0624c2
noisy_double(x) = ifelse.((@~ Bernoulli()), 2*x, x)

# ╔═╡ 0731d3bb-c8bc-4ef3-9ce0-e5f1290717ce
randsample(noisy_double(3))

# ╔═╡ 8dffbe36-0342-45a1-8418-5bbfd140eed9
md"## Example: Causal Models in Medical Diagnosis"

# ╔═╡ f4607433-ffe8-4bed-a1b9-40ee7288d299
md"Generative knowledge is often causal knowledge that describes how events or states of the world are related to each other. As an example of how causal knowledge can be encoded in Omega, consider a simplified medical scenario:"

# ╔═╡ 20179d7a-b48a-4c21-b490-d63153a20e3d
let
	lung_cancer = @~ Bernoulli(0.01)
	cold = @~ Bernoulli(0.2)
	cough = cold .| lung_cancer
	randsample(cough)
end

# ╔═╡ af236993-8e61-4ab8-91e9-2fa2485419d0
md"This program models the diseases and symptoms of a patient in a doctor’s office. It first specifies the base rates of two diseases the patient could have: lung cancer is rare while a cold is common, and there is an independent chance of having each disease. The program then specifies a process for generating a common symptom of these diseases – an effect with two possible causes: The patient coughs if they have a cold or lung cancer (or both).
Here is a more complex version of this causal model:"

# ╔═╡ fb12e17f-a21b-41f5-82fe-f75cbba3ed33
begin
	lung_cancer = @~ Bernoulli(0.01)
	TB = @~ Bernoulli(0.005)
	stomach_flu = @~ Bernoulli(0.1)
	cold = @~ Bernoulli(0.2)
	other = @~ Bernoulli(0.1)
end

# ╔═╡ 9250533e-7dc3-4d7e-afb6-62373895a7c3
cough = pw(|, 
	(cold .& (@~ Bernoulli())), 
	(lung_cancer .& (@~ Bernoulli(0.3))), 
	(TB .& (@~ Bernoulli(0.7))), 
	(other .& (@~ Bernoulli(0.1)))
)

# ╔═╡ e570ac58-b61f-4a26-bd24-5acd41f86140
fever = pw(|, 
	(cold .& (@~ Bernoulli(0.3))), 
	(stomach_flu .& (@~ Bernoulli())), 
	(TB .& (@~ Bernoulli(0.1))), 
	(other .& (@~ Bernoulli(0.01)))
)

# ╔═╡ 31e8ad83-6eb7-4cc3-83fd-308d119eb11d
chest_pain = pw(|, 
	(lung_cancer .& (@~ Bernoulli())), 
	(TB .& (@~ Bernoulli())), 
	(other .& (@~ Bernoulli(0.01)))
)

# ╔═╡ 1a7dde50-00dc-4b1b-99fc-315372a4fa53
shortness_of_breath = pw(|, 
	(lung_cancer .& (@~ Bernoulli())), 
	(TB .& (@~ Bernoulli(0.2))), 
	(other .& (@~ Bernoulli(0.01)))
)

# ╔═╡ 11b8a8b8-88e5-4485-a377-4d124472b6ba
symptoms = @joint cough fever chest_pain shortness_of_breath

# ╔═╡ 15073a61-fe7f-4bf4-942c-bd37ea30a59d
randsample(symptoms)

# ╔═╡ e5dcc735-3ea7-4d8f-b1c7-83c6a5959019
md"Now there are four possible diseases and four symptoms. Each disease causes a different pattern of symptoms. The causal relations are now probabilistic: Only some patients with a cold have a cough ($50\%$), or a fever ($30\%$). There is also a catch-all disease category “other”, which has a low probability of causing any symptom. Noisy logical functions—functions built from and (`&ₚ`), or (`|ₚ`), and distributions —provide a simple but expressive way to describe probabilistic causal dependencies between Boolean (true-false valued) variables.
When you run the above code, the program generates a list of symptoms for a hypothetical patient. Most likely all the symptoms will be false, as (thankfully) each of these diseases is rare. Experiment with running the program multiple times. Now try modifying the function for one of the diseases, setting it to be true, to simulate only patients known to have that disease. For example, replace `lung_cancer = @~ Bernoulli(0.01)` with `lung_cancer = true`. Run the program several times to observe the characteristic patterns of symptoms for that disease."

# ╔═╡ f56fc90c-abc5-4dfc-a127-5fff4bc48a36
md"## Persistent Randomness"

# ╔═╡ 19c97977-ee72-4cd9-b75a-f1f58a8a2518
md"In Omega, random variables are pure: reapplication to the same context (or ω) produces the same result."

# ╔═╡ d4c207f3-a6a2-4840-8bf2-d0c6b8153f8b
ω = defω()

# ╔═╡ df2062b1-e448-499b-b77b-eee111eb5ac3
f = 1 ~ Bernoulli()

# ╔═╡ c70202e9-a417-43cc-9ae6-960b25011127
g = 1 ~ Bernoulli()

# ╔═╡ 759a1b0d-2c2b-4b8f-ab0a-81a905d41518
f(ω) == g(ω) # Always returns true

# ╔═╡ 3a06f6c3-7cd0-4a54-94ea-68c54b7790b2
md"Independent random variables of a random class can be created in Omega by changing the `id` as follows -"

# ╔═╡ 5d5d87d4-0a55-4a4b-a47c-22443fbfccaf
let
	iid_1 = 1 ~ Bernoulli()
	iid_2 = 2 ~ Bernoulli()
 	ω = defω()
	iid_1(ω) == iid_2(ω) # Does not always return true
end

# ╔═╡ 0e7fbf3b-f646-44da-a9c4-6558e6da924f
md"Sometimes we require the results of the stochastic process to be random but persistent, for example: Eye colour of a person. We can represent the notion that eye color is random, but each person has a fixed eye colour as follows:"

# ╔═╡ aa35a2a6-c31f-4145-bec5-5237703de3fd
function eye_colour(n, ω)
	d = (n ~ DiscreteUniform(1, 3))(ω)
	if d == 1
		return :blue
	elseif d == 2
		return :green
	else
		return :brown
	end
end

# ╔═╡ 35fe9bcb-312e-4dce-a53c-d83019eb5c96
randsample(ω -> [eye_colour(:bob, ω), eye_colour(:alice, ω), eye_colour(:bob, ω)])

# ╔═╡ 6c8ee4bc-3d8b-4867-9258-ef3af9ab8b2d
md"Bob's eye colour is consistent every time we call the above `randsample`."

# ╔═╡ 753f852e-f33b-44bd-8d55-66560d752442
md"# Hypothetical Reasoning"

# ╔═╡ fa973d7b-bf48-47dd-bf76-bbba860620d3
md"Suppose that we know some fixed fact, and we wish to consider hypotheses about how a generative model could have given rise to that fact. In Omega, we can use conditional random variables to describe a distribution under some assumptions or conditions."

# ╔═╡ ab9689da-640f-4275-8096-fdd3cbcf8cca
md"Consider the following simple generative model:"

# ╔═╡ c79e3038-8834-4062-975c-caed9e2b8efc
A = @~ Bernoulli()

# ╔═╡ 023019ca-fd7c-4c2e-8f11-ab46a1373337
B = @~ Bernoulli()

# ╔═╡ 6d782392-ce54-4628-81b5-a856b7ce6853
C = @~ Bernoulli()

# ╔═╡ 1fa7f01f-495c-487c-8bbb-2850aea62215
model_sum = A .+ B .+ C

# ╔═╡ 7af3b2e4-1adf-41d8-a36e-148595e90612
histogram(randsample(model_sum, 1000))

# ╔═╡ e4898929-72fc-43ba-a539-768986136d3c
md"The process described in model samples three numbers and adds them. The value of the final expression here is $0$, $1$, $2$ or $3$. A priori, each of the variables `A`, `B`, `C` has $0.5$ probability of being $1$ or $0$. However, suppose that we know that the sum `model` is equal to $3$. How does this change the space of possible values that variable `A` could have taken? `A` (and `B` and `C`) must be _equal_ to $1$ for this result to happen. We can see this in the following Omega inference, where we use `|ᶜ` to express the desired assumption (ie., to condition on random variables):"

# ╔═╡ 5e98f646-208b-479c-a268-04a6a822f7b5
A_cnd = A |ᶜ (model_sum .== 3)

# ╔═╡ 3e21ca4c-30c0-4c14-aaeb-66296429117e
histogram(randsample(A_cnd, 100))

# ╔═╡ 5270564b-6359-44e0-9b32-06ff023046e2
md"""
The output here describes appropriate beliefs about the likely value of `A`, conditioned on `model` being equal to $3$.

Now suppose that we condition on `model` being greater than or equal to $2$. Then `A` need not be $1$, but it is more likely than not to be. The corresponding plot shows the appropriate distribution of beliefs for `A` conditioned on this new fact:
"""

# ╔═╡ 08310a02-e850-4234-9a85-958c8fb40904
A_cnd_new = A |ᶜ (model_sum .>= 2)

# ╔═╡ 93f3a800-08a1-4a0d-81e6-3b2453cbb912
histogram(randsample(A_cnd_new, 100))

# ╔═╡ 669db51a-d1dd-4c81-be84-9d76f1cdabe3
md"Getting back to the medical example above: "

# ╔═╡ 6f820403-f169-48d1-ac90-581fb93ca287
lung_cancer_cond = lung_cancer |ᶜ pw(&, cough, chest_pain, shortness_of_breath)

# ╔═╡ e92715be-42f2-492b-8aab-0893e9f5a4af
TB_cond = TB |ᶜ pw(&, cough, chest_pain, shortness_of_breath)

# ╔═╡ 69bf3131-5eb5-40d4-9730-72ae83931a0a
histogram(randsample(lung_cancer_cond, 1000))

# ╔═╡ 061bd103-6ff0-4b3a-867f-997c0538283b
histogram(randsample(TB_cond, 1000))

# ╔═╡ db45caf2-5f35-4873-87ba-20eec03c44bf
begin
	U1= @~ Bernoulli(1/2.0)
	U2= @~ Bernoulli(1/3.0)
	U3= @~ Bernoulli(1/3.0)
	X=U1 #exercise 
	W=ifelse.(X.==1,0,U2) # weight
	H=ifelse.(X.==1,0,U3) #   heart condition
end

# ╔═╡ de63254f-2b60-472d-aab1-7c24e4f7111b
histogram(randsample(H, 100))

# ╔═╡ 64b78ddc-de17-4901-92f9-0bfe3c4cf387
H_cond = H|ᶜ W.==1

# ╔═╡ 7153194f-f8d9-4ff6-8ffe-aa6a2539c08b
histogram(randsample(H_cond, 100))

# ╔═╡ 151f08ca-7bf1-4270-b7e3-ebff4110f2e8
md"You can change the sysmptoms (conditions) to see how the histgrams change."

# ╔═╡ 85bde745-a06f-4233-8aea-9aed2a059f99
H_inter = H |ᵈ (W=>1)

# ╔═╡ 4616ae16-d2f8-4e50-8adf-9a5c9db15fad
histogram(randsample(H_inter, 100))

# ╔═╡ 0c3cf324-176e-45f6-876f-4f505ec53893


# ╔═╡ b5c8a92e-afae-4406-b631-53f3c5569163
md"# Causal Dependence"

# ╔═╡ 4ad5405d-8caf-4a49-8f3f-3bcdc36c1a44
md"""
Let’s examine the notion of “causal dependence” a little more carefully. What does it mean to believe that $A$ depends causally on $B$? Viewing cognition through the lens of probabilistic programs, the most basic notions of causal dependence are in terms of the structure of the program and the flow of evaluation (or “control”) in its execution. We say that expression $A$ causally depends on expression $B$ if it is **necessary to evaluate $B$ in order to evaluate $A$**. (More precisely, the expression $A$ depends on expression $B$ if it is ever necessary to evaluate $B$ in order to evaluate $A$.) For instance, in this program `A` depends on `B` but not on `C` (the final expression depends on both `A` and `C`):
"""

# ╔═╡ 813658ab-be73-4375-a93a-131c067df761
let
	C = @~ Bernoulli()
	B = @~ Bernoulli()
	A = ifelse.(B, (@~ Bernoulli(0.1)), (@~ Bernoulli(0.4)))
	randsample(A .| C)
end

# ╔═╡ 55fabbcf-80e8-41b0-982b-965cb64399c1
md"""
For example, consider a simpler variant of our medical diagnosis scenario:
"""

# ╔═╡ 62ed6730-e59f-4052-8eba-eb22124b45bf
smokes_ = @~ Bernoulli(0.2)

# ╔═╡ 19955337-8932-493c-8ca4-19749c2235f9
lung_disease_ = (smokes_ .& (@~ Bernoulli(0.1))) .| (@~ Bernoulli(0.001))

# ╔═╡ 2c2ae9c1-fde2-4fca-a511-b7fc825f7105
cold_ = @~ Bernoulli(0.02)

# ╔═╡ 64101589-d9cf-41f6-886b-9476be8a41b4
cough_ = pw(|, 
	(cold_ .& @~ Bernoulli()), 
	(lung_disease_ .& @~ Bernoulli()), 
	@~ Bernoulli(0.001)
)

# ╔═╡ 02005fce-8700-4d89-afe4-911046a48f95
fever_ = (cold_ .& @~ Bernoulli()) .| @~ Bernoulli(0.01)

# ╔═╡ 5b937b43-1056-4489-be72-478457ac7812
chest_pain_ = (lung_disease_ .& @~ Bernoulli(0.2)) .| @~ Bernoulli(0.01)

# ╔═╡ e66a9a5f-2105-4bae-9005-213a9761a64d
shortness_of_breath_ = (lung_disease_ .& @~ Bernoulli(0.2)) .| @~ Bernoulli(0.01)

# ╔═╡ d15b8057-a187-458d-9fbd-bf0d53633ad3
cold_cond = cold_ |ᶜ cough_

# ╔═╡ 3020afab-7348-40e9-849f-83fe7a2e2a34
histogram(randsample(cold_cond, 1000))

# ╔═╡ 982d5a3f-226a-4260-9337-099b2ef974c3
lung_disease_cond = lung_disease_ |ᶜ cough_

# ╔═╡ 9d968c4a-6bc1-4dc1-9775-9645db082ec9
histogram(randsample(lung_disease_cond, 1000))

# ╔═╡ 01527577-e3a2-4099-90bb-cdf078c83e8e
md"""
Here, `cough_` depends causally on both `lung_disease_` and `cold_`, while `fever` depends causally on `cold_` but not `lung_disease_`. We can see that `cough_` depends causally on `smokes_` but only indirectly: although `cough_` does not call `smokes_` directly, in order to evaluate whether a patient coughs, we first have to evaluate the expression `lung_disease_` that must itself evaluate `smokes_`.
"""

# ╔═╡ 56ee77ea-b1b0-48e8-bdc1-8a7bca44238f
md"## Detecting Dependence Through Intervention"

# ╔═╡ 59b09c23-678d-433b-ad1d-b176e9739b95
md"""
The causal dependence structure is not always immediately clear from examining a program, particularly where there are complex functions calls. Another way to detect (or according to some philosophers, such as Jim Woodward, to _define_) causal dependence is more operational, in terms of “difference making”: If we manipulate $A$, does $B$ tend to change? By **_manipulate_ here we don’t mean an assumption in the sense of conditioning**. Instead we mean actually edit, or _intervene on_, the program in order to make an expression have a particular value independent of its (former) causes. If setting $A$ to different values in this way changes the distribution of values of $B$, then $B$ causally depends on $A$.
"""

# ╔═╡ 4523df4b-e72e-4a76-9187-a1b9f59bbd47
md"""
For example, in the above example of medical diagnosis, we now give our hypothetical patient a cold — say, by exposing him to a strong cocktail of cold viruses. We should not model this as an observation (e.g. by conditioning on having a cold), because we have taken direct action to change the normal causal structure. Instead, we implement intervention by directly editing the random variables:
"""

# ╔═╡ a0ace004-ae4f-4c6e-a176-da2a8c65ebe5
cough_intervened = cough_ |ᵈ (cold_ => true)

# ╔═╡ f28358df-c568-4363-af6d-c89504ef96cd
histogram(randsample(cough_intervened, 100))

# ╔═╡ 0d7b7227-73f1-4a6b-b477-ff30683e2ebe
md"""
You should see that the distribution on `cough_` changes: coughing becomes more likely if we know that a patient has been given a cold by external intervention. But the reverse is not true: Try forcing the patient to have a cough (e.g., with some unusual drug or by exposure to some cough-inducing dust) by writing `cough_ => true` instead of `cold_ => true`: the distribution on `cold_` is unaffected. We have captured a familiar fact: treating the symptoms of a disease directly doesn’t cure the disease (taking cough medicine doesn’t make your cold go away), but treating the disease _does_ relieve the symptoms.

Verify in the program above that the method of manipulation works also to identify causal relations that are only indirect: for example, force a patient to smoke and show that it increases their probability of coughing, but not vice versa.

If we are given a program representing a causal model, and the model is simple enough, it is straightforward to read off causal dependencies from the program code itself. However, the notion of causation as difference-making may be easier to compute in much larger, more complex models—and it does not require an analysis of the program code. As long as we can modify (or imagine modifying) the definitions in the program and can run the resulting model, we can compute whether two events or functions are causally related by the difference-making criterion.
"""

# ╔═╡ 64f78304-ac14-42bb-bbe8-0c2e4665ff35
md"""
# From _A Priori_ Dependence to Conditional Dependence

The relationships between causal structure and statistical dependence become particularly interesting and subtle when we look at the effects of additional observations or assumptions. Events that are statistically dependent a priori may become independent when we condition on some observation; this is called screening off. Also, events that are statistically independent a priori may become dependent when we condition on observations; this is known as explaining away. The dynamics of screening off and explaining away are extremely important for understanding patterns of inference—reasoning and learning—in probabilistic models.
"""

# ╔═╡ 93a2b5e2-d45e-43fe-8ed4-f3244fd0bfae
md"""
Continuing with the medical diagnosis example, in the model `smokes` is statistically dependent on several symptoms— `cough`, `chest_pain`, and `shortness_of_breath` —due to a causal chain between them mediated by `lung_disease`. We can see this easily by conditioning on these symptoms and looking at `smokes`:
"""

# ╔═╡ 87d3eb73-4d13-4127-9c00-ea85c01d2abc
smokes_cond_c_cp_sob = smokes_ |ᶜ .&(cough_, chest_pain_, shortness_of_breath_)

# ╔═╡ a78f84b2-19e3-4464-a936-cde802f3e9c0
histogram(randsample(smokes_cond_c_cp_sob, 1000))

# ╔═╡ f75967b2-0720-4d24-972b-fcd8fcdf89a7
md"""
The conditional probability of smokes is much higher than the base rate, $0.2$, because observing all these symptoms gives strong evidence for smoking. See how much evidence the different symptoms contribute by dropping them out of the conditioning set. (For instance, try conditioning on `cough &ₚ chest_pain`, or just `cough`; you should observe the probability of `smokes` decrease as fewer symptoms are observed.)

Now, suppose we condition also on knowledge about the function that mediates these causal links: `lung_disease`. Is there still an informational dependence between these various symptoms and `smokes`? In the Inference below, try adding and removing various symptoms (`cough`, `chest_pain`, `shortness_of_breath`) but maintaining the observation `lung_disease`:
"""

# ╔═╡ 4d08416a-bcdf-4cc0-a57e-9f47ce75d720
smokes_cond_c_cp_sob_ld = 
	smokes_ |ᶜ .&(lung_disease_, cough_, chest_pain_, shortness_of_breath_)

# ╔═╡ af88d1b7-233e-4c24-ae4c-18b8ab4f8fe9
md"""
You should see an effect of whether the patient has lung disease on conditional inferences about smoking—a person is judged to be substantially more likely to be a smoker if they have lung disease than otherwise—but there are no separate effects of chest pain, shortness of breath, or cough over and above the evidence provided by knowing whether the patient has lung-disease. The intermediate variable lung disease screens off the root cause (smoking) from the more distant effects (coughing, chest pain and shortness of breath).

Here is a concrete example of explaining away in our medical scenario. Having a cold and having lung disease are a priori independent both causally and statistically. But because they are both causes of coughing if we observe `cough` then `cold` and `lung_disease` become statistically dependent. That is, learning something about whether a patient has `cold` or `lung_disease` will, in the presence of their common effect `cough`, convey information about the other condition. `cold` and `lung_disease` are a priori independent, but conditionally dependent given `cough`.

To illustrate, observe how the probabilities of `cold` and `lung_disease` change when we observe `cough` is `true`:
"""

# ╔═╡ 19e0a525-4f2a-40e6-8e1b-fcc5b08a3625
histogram(randsample(smokes_cond_c_cp_sob_ld, 1000))

# ╔═╡ ead92b5d-c691-4034-b338-776d415b1cf1
ld_and_cold = @joint lung_disease_ cold_

# ╔═╡ ccf3d869-f36c-4c70-8a6b-440ed1c078a5
ld_and_cold_cond = ld_and_cold |ᶜ cough_

# ╔═╡ 09d69f67-7fe3-4305-8e21-db299acd7c31
cond_cough_samples = randsample(ld_and_cold_cond, 1000)

# ╔═╡ a85d8357-5888-481c-b532-3c61dd26f3dc
barplot(Dict(freqtable(cond_cough_samples)), xlabel = "Frequency")

# ╔═╡ 440d2ee4-514e-4e16-8228-1d4afb2ac49c
barplot(Dict(freqtable(map(u -> string(u[1]), cond_cough_samples))), title = "Lung Disease")

# ╔═╡ 33ad29dc-50c2-4c3a-aede-4e99ce2ba982
barplot(Dict(freqtable(map(u -> string(u[2]), cond_cough_samples))), title = "Cold")

# ╔═╡ 9b0932af-35e5-4d1b-a341-7a8d0ec374d4
md"""
Both cold and lung disease are now far more likely than their baseline probability: the probability of having a cold increases from $2\%$ to around $50\%$; the probability of having lung disease also increases from $2.1\%$ to around $50\%$.

Now suppose we also learn that the patient does _not_ have a cold.
"""

# ╔═╡ 7829c5f2-6a3c-447a-b789-d34e5e29b191
cond_cough_not_cold = ld_and_cold |ᶜ (cough_ .& .!(cold_))

# ╔═╡ 2d9d5a21-b7d5-40a9-9fbc-59c0408f77e9
cond_cough_not_cold_samples = randsample(cond_cough_not_cold, 1000)

# ╔═╡ 1b256345-9347-4488-929b-4ef7eb8cce5f
barplot(Dict(freqtable(cond_cough_not_cold_samples)), xlabel = "Frequency")

# ╔═╡ 9fa1394a-9e69-4a4f-b0c6-4cab7c81dd74
barplot(Dict(freqtable(map(u -> string(u[1]), cond_cough_not_cold_samples))), title = "Lung Disease")

# ╔═╡ 2e12872e-2383-4bf6-80fa-ac6a5e7af230
barplot(Dict(freqtable(map(u -> string(u[2]), cond_cough_not_cold_samples))), title = "Cold")

# ╔═╡ f6059bd4-d73d-4ad7-9cfd-8e00f783353d
md"""
The probability of having lung disease increases dramatically. If instead we had observed that the patient does have a cold, the probability of lung cancer returns to its base rate of $2.1\%$
"""

# ╔═╡ e49c8354-ee19-4c38-ba2a-22086c031ec3
cond_cough_and_cold = ld_and_cold |ᶜ (cough_ .& cold_)

# ╔═╡ 4d716eb8-b5c8-4a21-beea-12b075902de7
cond_cough_cold_samples = randsample(cond_cough_and_cold, 1000)

# ╔═╡ 2777347b-0a1d-4257-a74b-72b83495a20a
barplot(Dict(freqtable(cond_cough_cold_samples)), xlabel = "Frequency")

# ╔═╡ 85c8d305-09a7-4947-9a62-47000d48b573
barplot(Dict(freqtable(map(u -> string(u[1]), cond_cough_cold_samples))), title = "Lung Disease")

# ╔═╡ 10c7492f-c9f5-4a2f-a1bc-a3c00e8eec7c
barplot(Dict(freqtable(map(u -> string(u[2]), cond_cough_cold_samples))), title = "Cold")

# ╔═╡ 3547bd80-2e4f-4b81-92b9-5f1fa28ee7c4
md"""
This is the conditional statistical dependence between lung disease and cold, given cough: Learning that the patient does in fact have a cold “explains away” the observed cough, so the alternative of lung disease decreases to a much lower value — roughly back to its $1$ in a $1000$ rate in the general population. If on the other hand, we had learned that the patient does not have a cold, so the most likely alternative to lung disease is not in fact available to “explain away” the observed cough, which raises the conditional probability of lung disease dramatically. As an exercise, check that if we remove the observation of coughing, the observation of having a cold or not has no influence on our belief about lung disease; this effect is purely conditional on the observation of a common effect of these two causes.

Explaining away effects can be more indirect. Instead of observing the truth value of cold, a direct alternative cause of cough, we might simply observe another symptom that provides evidence for cold, such as fever. Compare these conditions using the above program to see an “explaining away” conditional dependence in belief between `fever` and `lung_disease`.
"""

# ╔═╡ 3b7d4453-ac73-41cc-8b17-b41dd5993f62
cough_not_cold_samples = randsample(ld_and_cold, 10000)

# ╔═╡ 6d79e7cf-7a66-4b93-95cb-489eef37b160
barplot(Dict(freqtable(cough_not_cold_samples
)), xlabel = "Frequency")

# ╔═╡ fceff4e5-aa7a-433c-afde-0f6f1f95cf13
lung_disease_

# ╔═╡ 82ee4da4-16de-486f-8485-4c97f2db8984
md"""
# Bayesian Data Analysis
"""

# ╔═╡ e2bdffbf-9015-45af-88ac-a21efdc1454d
md"""
Bayesian data analysis (BDA) is a general-purpose approach to making sense of data. A BDA model is an explicit hypotheses about the generative process behind the experimental data – where did the observed data come from? For instance, the hypothesis that data from two experimental conditions came from two _different_ distributions. After making explicit hypotheses, Bayesian inference can be used to invert the model: go from experimental data to updated beliefs about the hypotheses.
"""

# ╔═╡ 82eb4aed-2df4-49fd-b0b5-1137792a9eb9
md"""
### Parameters and predictives
In a BDA model the random variables are usually called parameters. Parameters can be of theoretical interest, or not (the latter are called nuisance parameters). Parameters are in general unobservable (or, “latent”), so we must infer them from observed data. We can also go from updated beliefs about parameters to expectations about future data, so call _posterior predictives_.

For a given Bayesian model (together with data), there are four conceptually distinct distributions we often want to examine. For parameters, we have priors and posteriors:
* The _prior distribution_ over parameters captures our initial state of knowledge (or beliefs) **about the values that the latent parameters could have, before seeing the data**.
* The _posterior distribution_ over parameters captures what we know about the latent parameters having updated our beliefs with the evidence provided by data.

From either the prior or the posterior over parameters we can then run the model forward, to get predictions about data sets:

* The **prior predictive&& distribution tells us what data to expect, given our model and our initial beliefs about the parameters. The prior predictive is a distribution over data, and gives the relative probability of different observable outcomes before we have seen any data.
* The **posterior predictive** distribution tells us what data to expect, given the same model we started with, but with beliefs that have been updated by the observed data. The posterior predictive is a distribution over data, and gives the relative probability of different observable outcomes, after some data has been seen.
Loosely speaking, _predictive_ distributions are in “data space” and _parameter_ distributions are in “latent parameter space”.

### Example: Election surveys
Imagine you want to find out how likely Candidate A is to win an election. To do this, you will try to estimate the proportion of eligible voters who will vote for Candidate A in the election. Trying to determine directly how many (voting age, likely to vote) people prefer Candidate A vs. Candidate B would require asking over 100 million people. It’s impractical to measure the whole distribution. Instead, pollsters measure a sample (maybe ask 1000 people), and use that to draw conclusions about the “true population proportion” (an unobservable parameter).

Here, we explore the result of an experiment with 20 trials and binary outcomes (“will you vote for Candidate A or Candidate B?”).
"""

# ╔═╡ 584ee043-7afd-4f3a-b627-8334f644772d
begin
	## observed data
	k = 1  # number of people who support candidate A
    n = 20 # number of people asked
end

# ╔═╡ 41f13cec-ef9e-46e1-8195-3a93473c4d43
p = @~ Uniform(0, 1) # true population proportion who support candidate A

# ╔═╡ 599155d9-62a6-417a-ad6e-7672b2cb2046
# recreate model structure, without conditioning
prior_predictive = @~ Binomial.(n, p)

# ╔═╡ 733a1296-c9c4-497b-9ce7-54584f5ab82c
histogram(randsample(prior_predictive, 1000))

# ╔═╡ 44edc3d8-1cbc-414a-a31c-78186696bc4a
# Observed k people support "A" 
# Assuming each person's response is independent of each other
posterior_predictive = p |ᶜ (ω::Ω -> (@~ Binomial(n, p(ω)))(ω) .== k)

# ╔═╡ 3983e58e-6e6c-401c-87da-e646d9174359
posterior_predictive_samples = randsample(posterior_predictive, 1000)

# ╔═╡ f784cef0-8e39-4793-8780-00a152680c62
histogram(posterior_predictive_samples)

# ╔═╡ 555d26b3-3737-4770-8cfd-f6a4fafb07bc
md"""
What can we conclude intuitively from examining these plots? First, because prior differs from posterior, the evidence has changed our beliefs. Second, the posterior predictive assigns quite high probability to the true data, suggesting that the model considers the data “reasonable”. Finally, after observing the data, we conclude the true proportion of people supporting Candidate A is quite low – around $0.09$, or anyhow somewhere between $0.0$ and $0.15$. Check your understanding by trying other data sets, varying both `k` and `n`.
"""

# ╔═╡ e7c7937b-7222-42f9-980c-ebc41279d5d5
md"""
### Quantifying claims about parameters

How can we quantify a claim like “the true parameter is low”? One possibility is to compute the mean or expected value of the parameter, which is mathematically given by $∫x⋅p(x)dx$ for a posterior distribution $p(x)$. Thus in the above election example we could:
"""

# ╔═╡ 4e8a63ed-bd6e-47a6-ae78-149a7e2726fa
mean(posterior_predictive_samples)

# ╔═╡ 67bfa2a7-bdca-4ccd-b887-1f9e753f104a
md"""
This tells us that the mean is about $0.09$. This can be a very useful way to summarize a posterior, but it eliminates crucial information about how _confident_ we are in this mean. A coherent way to summarize our confidence is by exploring the probability that the parameter lies within a given interval. Conversely, an interval that the parameter lies in with high probability (say $90\%$) is called a _credible interval_ (CI). Let’s explore credible intervals for the parameter in the above model:
"""

# ╔═╡ f14dfca1-5753-4685-9ce0-3df7f0c7fe92
mean(0.01 .< posterior_predictive_samples .< 0.18)

# ╔═╡ 842b3cbc-8705-42e2-8612-fe173f798b2f
md"""
Here we see that $[0.01, 0.18]$ is an (approximately) $90\%$ credible interval – we can be about $90\%$ sure that the true parameter lies within this interval. Notice that the $90\%$ CI is not unique. There are different ways to choose a particular CI. One particularly common, and useful, one is the Highest Density Interval (HDI), which is the smallest interval achieving a given confidence. (For unimodal distributions the HDI is unique and includes the mean.)
"""

# ╔═╡ d98c8a13-f771-4a90-93ab-77285595664b
md"""
### Model selection
In the above examples, we’ve had a single data-analysis model and used the experimental data to learn about the parameters of the models and the descriptive adequacy of the models. Often as scientists, we are in fortunate position of having multiple, distinct models in hand, and want to decide if one or another is a better description of the data. The problem of _model selection_ given data is one of the most important and difficult tasks of BDA.

Imagine in the above example we begin with a (rather unhelpful) data analysis model that assumes each candidate is equally likely to win, that is `p=0.5`. We quickly notice, by looking at the posterior predictives, that this model doesn’t accommodate the data well at all. We thus introduce the above model where `p = @~ Uniform(0,1)`. How can we quantitatively decide which model is better? One approach is to combine the models into an uber model that decides which approach to take:
"""

# ╔═╡ ff83b88d-cfb7-4d76-a4b6-d239921c12b4
begin
	# observed data
	k_ = 5 # number of people who support candidate A
	n_ = 20  # number of people asked
end

# ╔═╡ 4ac4efca-80c6-49e9-bb06-01269704ce52
# binary decision variable for which hypothesis is better
d = Variable(ω -> (@~ Bernoulli())(ω) ? "simple" : "complex")

# ╔═╡ 69683817-1f2d-4e3f-a5c2-211ebab1e8c6
p_ = Variable(ω -> (d(ω) == "simple") ? 0.5 : (@~ Uniform())(ω))

# ╔═╡ f8e876fb-9d83-48d8-b138-af750cd295e6
posterior_ = d |ᶜ ((@~ Binomial.(n_, p_)) .== k_)

# ╔═╡ da7aee56-8946-4600-8e9f-3a21c0a8754f
barplot(Dict(freqtable(randsample(posterior_, 1000))))

# ╔═╡ 94efc0e3-fcfc-4492-8e2e-9b72abe48b84
md"""
We see that, as expected, the more complex model is preferred: we can confidently say that given the data the more complex model is the one we should believe. Further we can quantify this via the posterior probability of the complex model.

This model is an example from the classical hypothesis testing framework. We consider a model that fixes one of its parameters to a pre-specified value of interest (here $H₀:p=0.5$). This is sometimes referred to as a _null hypothesis_. The other model says that the parameter is free to vary. In the classical hypothesis testing framework, we would write: $H₁:p≠0.5$. With Bayesian hypothesis testing, we must be explicit about what $p$ is (not just what $p$ is not), so we write $H₁:p∼Uniform(0,1)$.
"""

# ╔═╡ 4a1414fe-9003-47ee-b834-c92018694a09
barplot(Dict(freqtable(cond_cough_samples)), xlabel = "Frequency")

# ╔═╡ 6ba1668d-0085-407a-b97f-afc900f95db2


# ╔═╡ Cell order:
# ╠═e0b80775-ca75-4d05-9863-a534c27f47ff
# ╟─f8041cae-ef14-11ec-3bf2-67b19132ea93
# ╟─d85d114a-be83-4ee6-a4c3-136b34c82112
# ╟─fda57285-7d1d-4618-a0e4-8f1946936115
# ╟─528c88c9-a731-4628-b01b-7b6807847bfd
# ╠═0f085170-058a-45df-b377-1616c52c7875
# ╟─b08ac749-f261-4b3a-b269-759351025bd5
# ╠═6b2c770c-1dac-4a06-bdbd-585a194203e3
# ╠═bdd7cb5f-1e11-4886-ad82-122e2ae598a7
# ╟─2e89bd6f-1ccf-4440-88fa-157d06121713
# ╟─713af13a-e328-4719-9be2-b1baf181ae25
# ╟─4fcea488-67d0-4a7f-8411-56f0db1c5713
# ╠═989c9194-ed60-4525-a371-f5d63c8dfa89
# ╟─2e17790c-651a-4546-88e8-00dfd9bcfacc
# ╟─1a31c200-569f-46d1-8cd2-ee3ce5008ac1
# ╟─508eaa3b-7146-437c-9efe-f493f28bd81b
# ╟─7b111711-cbb5-40c8-89bd-e54f3855df54
# ╠═ba91848f-1abb-4728-ba97-dcdc380a7e69
# ╠═0091feef-8c36-4d72-aa93-3a97c7265dc4
# ╠═557d9b90-769e-4ab0-8f22-172c7ce83cdf
# ╟─5e41d415-349e-416c-b30a-80ff38b60011
# ╠═4eae034d-2205-4b7d-acae-92d99868bf19
# ╠═abec0967-0b47-49e5-9678-aa64959f12d9
# ╠═f7274e22-ab59-4fdb-96a9-73873e1ef283
# ╟─580d51dd-5eb0-48b1-9827-aa036328b925
# ╠═075533af-bff9-453c-91ab-e5c24c8aed9d
# ╟─69bde3f4-d211-4185-ade2-ffc44893aa56
# ╠═f526ddd6-ae51-4ff0-8c07-0098c2253d66
# ╠═fad059c0-484c-4abb-bc82-e46442ec98c1
# ╟─2dd0abb4-6692-4b89-810d-aeed7cbd3e05
# ╟─41f297fd-3b53-412d-8ab4-8560c44c1229
# ╠═4f2bef6a-51de-4f44-9ba1-ad3fbee0f722
# ╟─2cab677e-ff84-4d0e-a880-7b8fce90b147
# ╠═0c8164ae-20e2-4590-8c9a-2e0f25d28505
# ╟─5d8e4aba-1a30-4212-8d80-10036faacdf7
# ╠═11d7f790-b50a-4d89-bd8d-b08db06b5056
# ╟─86a7ce00-c37a-431b-8121-e5464fb0d4b3
# ╠═48c9cbfe-07f9-4b6f-903e-03888a0624c2
# ╠═0731d3bb-c8bc-4ef3-9ce0-e5f1290717ce
# ╟─8dffbe36-0342-45a1-8418-5bbfd140eed9
# ╟─f4607433-ffe8-4bed-a1b9-40ee7288d299
# ╠═20179d7a-b48a-4c21-b490-d63153a20e3d
# ╟─af236993-8e61-4ab8-91e9-2fa2485419d0
# ╠═fb12e17f-a21b-41f5-82fe-f75cbba3ed33
# ╠═9250533e-7dc3-4d7e-afb6-62373895a7c3
# ╠═e570ac58-b61f-4a26-bd24-5acd41f86140
# ╠═31e8ad83-6eb7-4cc3-83fd-308d119eb11d
# ╠═1a7dde50-00dc-4b1b-99fc-315372a4fa53
# ╠═11b8a8b8-88e5-4485-a377-4d124472b6ba
# ╠═15073a61-fe7f-4bf4-942c-bd37ea30a59d
# ╟─e5dcc735-3ea7-4d8f-b1c7-83c6a5959019
# ╟─f56fc90c-abc5-4dfc-a127-5fff4bc48a36
# ╟─19c97977-ee72-4cd9-b75a-f1f58a8a2518
# ╠═d4c207f3-a6a2-4840-8bf2-d0c6b8153f8b
# ╠═df2062b1-e448-499b-b77b-eee111eb5ac3
# ╠═c70202e9-a417-43cc-9ae6-960b25011127
# ╠═759a1b0d-2c2b-4b8f-ab0a-81a905d41518
# ╟─3a06f6c3-7cd0-4a54-94ea-68c54b7790b2
# ╠═5d5d87d4-0a55-4a4b-a47c-22443fbfccaf
# ╟─0e7fbf3b-f646-44da-a9c4-6558e6da924f
# ╠═aa35a2a6-c31f-4145-bec5-5237703de3fd
# ╠═35fe9bcb-312e-4dce-a53c-d83019eb5c96
# ╟─6c8ee4bc-3d8b-4867-9258-ef3af9ab8b2d
# ╟─753f852e-f33b-44bd-8d55-66560d752442
# ╟─fa973d7b-bf48-47dd-bf76-bbba860620d3
# ╟─ab9689da-640f-4275-8096-fdd3cbcf8cca
# ╠═c79e3038-8834-4062-975c-caed9e2b8efc
# ╠═023019ca-fd7c-4c2e-8f11-ab46a1373337
# ╠═6d782392-ce54-4628-81b5-a856b7ce6853
# ╠═1fa7f01f-495c-487c-8bbb-2850aea62215
# ╠═7af3b2e4-1adf-41d8-a36e-148595e90612
# ╟─e4898929-72fc-43ba-a539-768986136d3c
# ╠═5e98f646-208b-479c-a268-04a6a822f7b5
# ╠═3e21ca4c-30c0-4c14-aaeb-66296429117e
# ╟─5270564b-6359-44e0-9b32-06ff023046e2
# ╠═08310a02-e850-4234-9a85-958c8fb40904
# ╠═93f3a800-08a1-4a0d-81e6-3b2453cbb912
# ╟─669db51a-d1dd-4c81-be84-9d76f1cdabe3
# ╠═6f820403-f169-48d1-ac90-581fb93ca287
# ╠═e92715be-42f2-492b-8aab-0893e9f5a4af
# ╠═69bf3131-5eb5-40d4-9730-72ae83931a0a
# ╠═061bd103-6ff0-4b3a-867f-997c0538283b
# ╠═db45caf2-5f35-4873-87ba-20eec03c44bf
# ╠═de63254f-2b60-472d-aab1-7c24e4f7111b
# ╠═64b78ddc-de17-4901-92f9-0bfe3c4cf387
# ╠═7153194f-f8d9-4ff6-8ffe-aa6a2539c08b
# ╟─151f08ca-7bf1-4270-b7e3-ebff4110f2e8
# ╠═85bde745-a06f-4233-8aea-9aed2a059f99
# ╠═4616ae16-d2f8-4e50-8adf-9a5c9db15fad
# ╠═0c3cf324-176e-45f6-876f-4f505ec53893
# ╟─b5c8a92e-afae-4406-b631-53f3c5569163
# ╟─4ad5405d-8caf-4a49-8f3f-3bcdc36c1a44
# ╠═813658ab-be73-4375-a93a-131c067df761
# ╟─55fabbcf-80e8-41b0-982b-965cb64399c1
# ╠═62ed6730-e59f-4052-8eba-eb22124b45bf
# ╠═19955337-8932-493c-8ca4-19749c2235f9
# ╠═2c2ae9c1-fde2-4fca-a511-b7fc825f7105
# ╠═64101589-d9cf-41f6-886b-9476be8a41b4
# ╠═02005fce-8700-4d89-afe4-911046a48f95
# ╠═5b937b43-1056-4489-be72-478457ac7812
# ╠═e66a9a5f-2105-4bae-9005-213a9761a64d
# ╠═d15b8057-a187-458d-9fbd-bf0d53633ad3
# ╠═3020afab-7348-40e9-849f-83fe7a2e2a34
# ╠═982d5a3f-226a-4260-9337-099b2ef974c3
# ╠═9d968c4a-6bc1-4dc1-9775-9645db082ec9
# ╟─01527577-e3a2-4099-90bb-cdf078c83e8e
# ╟─56ee77ea-b1b0-48e8-bdc1-8a7bca44238f
# ╟─59b09c23-678d-433b-ad1d-b176e9739b95
# ╟─4523df4b-e72e-4a76-9187-a1b9f59bbd47
# ╠═a0ace004-ae4f-4c6e-a176-da2a8c65ebe5
# ╠═f28358df-c568-4363-af6d-c89504ef96cd
# ╟─0d7b7227-73f1-4a6b-b477-ff30683e2ebe
# ╟─64f78304-ac14-42bb-bbe8-0c2e4665ff35
# ╟─93a2b5e2-d45e-43fe-8ed4-f3244fd0bfae
# ╠═87d3eb73-4d13-4127-9c00-ea85c01d2abc
# ╠═a78f84b2-19e3-4464-a936-cde802f3e9c0
# ╟─f75967b2-0720-4d24-972b-fcd8fcdf89a7
# ╠═4d08416a-bcdf-4cc0-a57e-9f47ce75d720
# ╟─af88d1b7-233e-4c24-ae4c-18b8ab4f8fe9
# ╠═19e0a525-4f2a-40e6-8e1b-fcc5b08a3625
# ╠═ead92b5d-c691-4034-b338-776d415b1cf1
# ╠═ccf3d869-f36c-4c70-8a6b-440ed1c078a5
# ╠═09d69f67-7fe3-4305-8e21-db299acd7c31
# ╠═a85d8357-5888-481c-b532-3c61dd26f3dc
# ╠═440d2ee4-514e-4e16-8228-1d4afb2ac49c
# ╠═33ad29dc-50c2-4c3a-aede-4e99ce2ba982
# ╟─9b0932af-35e5-4d1b-a341-7a8d0ec374d4
# ╠═7829c5f2-6a3c-447a-b789-d34e5e29b191
# ╠═2d9d5a21-b7d5-40a9-9fbc-59c0408f77e9
# ╠═1b256345-9347-4488-929b-4ef7eb8cce5f
# ╠═9fa1394a-9e69-4a4f-b0c6-4cab7c81dd74
# ╠═2e12872e-2383-4bf6-80fa-ac6a5e7af230
# ╟─f6059bd4-d73d-4ad7-9cfd-8e00f783353d
# ╠═e49c8354-ee19-4c38-ba2a-22086c031ec3
# ╠═4d716eb8-b5c8-4a21-beea-12b075902de7
# ╠═2777347b-0a1d-4257-a74b-72b83495a20a
# ╠═85c8d305-09a7-4947-9a62-47000d48b573
# ╠═10c7492f-c9f5-4a2f-a1bc-a3c00e8eec7c
# ╟─3547bd80-2e4f-4b81-92b9-5f1fa28ee7c4
# ╠═3b7d4453-ac73-41cc-8b17-b41dd5993f62
# ╠═6d79e7cf-7a66-4b93-95cb-489eef37b160
# ╠═fceff4e5-aa7a-433c-afde-0f6f1f95cf13
# ╟─82ee4da4-16de-486f-8485-4c97f2db8984
# ╟─e2bdffbf-9015-45af-88ac-a21efdc1454d
# ╟─82eb4aed-2df4-49fd-b0b5-1137792a9eb9
# ╠═584ee043-7afd-4f3a-b627-8334f644772d
# ╠═41f13cec-ef9e-46e1-8195-3a93473c4d43
# ╠═599155d9-62a6-417a-ad6e-7672b2cb2046
# ╠═733a1296-c9c4-497b-9ce7-54584f5ab82c
# ╠═44edc3d8-1cbc-414a-a31c-78186696bc4a
# ╠═3983e58e-6e6c-401c-87da-e646d9174359
# ╠═f784cef0-8e39-4793-8780-00a152680c62
# ╟─555d26b3-3737-4770-8cfd-f6a4fafb07bc
# ╟─e7c7937b-7222-42f9-980c-ebc41279d5d5
# ╠═4e8a63ed-bd6e-47a6-ae78-149a7e2726fa
# ╟─67bfa2a7-bdca-4ccd-b887-1f9e753f104a
# ╠═f14dfca1-5753-4685-9ce0-3df7f0c7fe92
# ╟─842b3cbc-8705-42e2-8612-fe173f798b2f
# ╟─d98c8a13-f771-4a90-93ab-77285595664b
# ╠═ff83b88d-cfb7-4d76-a4b6-d239921c12b4
# ╠═4ac4efca-80c6-49e9-bb06-01269704ce52
# ╠═69683817-1f2d-4e3f-a5c2-211ebab1e8c6
# ╠═f8e876fb-9d83-48d8-b138-af750cd295e6
# ╠═da7aee56-8946-4600-8e9f-3a21c0a8754f
# ╟─94efc0e3-fcfc-4492-8e2e-9b72abe48b84
# ╠═4a1414fe-9003-47ee-b834-c92018694a09
# ╠═6ba1668d-0085-407a-b97f-afc900f95db2
