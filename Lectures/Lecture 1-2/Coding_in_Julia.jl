### A Pluto.jl notebook ###
# v0.18.4

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

# ╔═╡ 97b02a67-4eb2-4374-8939-f947eeddf637
using Memoize

# ╔═╡ 66ffb748-e9b6-4833-a684-c7ccce08b622
using Random

# ╔═╡ fb324892-8995-4c51-9d6c-0f4eea2ed1e9
using Distributions

# ╔═╡ 189ed23d-5903-4461-aea2-17183939b1f9
using LinearAlgebra

# ╔═╡ dd3c6efd-9344-4a52-a681-59922f3bac6f
using PlutoUI

# ╔═╡ abd26020-9fb9-11ec-162c-571e629a4da6
md"# **Programming in Julia**"

# ╔═╡ 3ed99cc3-cf30-48df-9515-f9359d9c2c92
md"Machine Learning models are capable of learning from data. These models are usually implemented using one or more programming language. Hence data scientists have to have knowledge of one or more programming languages. In this section we introduce the Julia Programming Language as a valuable tool for the data scientist."

# ╔═╡ da931c4d-149a-4bb1-82b3-7a7ae3852f32
begin
	__JuliaString = html"<img src =https://julialang.org/assets/infra/logo.svg alt=Julia width=60px>" 

	md"""
	## What is $(__JuliaString)?
	
	Julia is a high level, high performance, general purpose programming language, designed at the MIT lab by Jeff Bezanson, Alan Edelman, Stefan Karpinski, Viral B. Shah with it's first public release made in 2012.
	
	According to the developers in their blogpost [Why We Created Julia](https://julialang.org/blog/2012/02/why-we-created-julia/):
	
	> We want a language that's open source, with a liberal license. We want the speed of C with the dynamism of Ruby. We want a language that's homoiconic, with true macros like Lisp, but with obvious, familiar mathematical notation like Matlab. We want something as usable for general programming as Python, as easy for statistics as R, as natural for string processing as Perl, as powerful for linear algebra as Matlab, as good at gluing programs together as the shell. Something that is dirt simple to learn, yet keeps the most serious hackers happy. We want it interactive and we want it compiled. (Did we mention it should be as fast as C?)
	
	Julia has since then grown tremendously and is currently in her 6th Stable release, with the language being popular among mathematicians, data scientists, economists, engineers, climate modellers etc. 
	"""
end

# ╔═╡ fea89255-14da-4733-abc4-f34f9518440d
md"""
## Why use Julia?
1. High Level Syntax. (Descriptive programming like syntax). Easy Syntax to learn.
2. Speed comparable to C and Fortran.
3. Great Package Manager
4. C/Fortran/Python Interoperability.
5. Fast growing community
6. Dynamic Type System
7. Easy Parallelization
8. Solves the two Language Problem.
"""

# ╔═╡ 20822d76-6e88-4051-b0c9-eaca30d50a96
md"""
# Julia Basics

## **Julia as a calculator**
"""

# ╔═╡ 3c439fc7-29cc-45db-8a7c-15ccc1139a31
sum(rand(3, 3))

# ╔═╡ 14753e06-0cdb-4314-8ef3-c04d29ca3495
sin(20)^2 + cos(20)^2

# ╔═╡ 686a0c83-842e-4c9b-a672-bb03ba2599e6
test=5

# ╔═╡ 66a7b5f6-d38b-4074-88db-32cfb57d2e27
test

# ╔═╡ 117cbe3b-b417-4560-ba57-f03c5f2db884


# ╔═╡ 23ddab9f-d996-4ecf-860a-7cc5412b226c
md"""
## **Variables and Assignments**

Rules for determining valid variable names in Julia are similar to that of Python or  R. Julia also allows programmers to use a vast range of unicode characters (μ, ϕ, etc.) as variable names, making mathematical code easier to read. For the detailed rules for variable names in julia, see [Allowed Varible Names](https://docs.julialang.org/en/v1/manual/variables/#man-allowed-variable-names).
"""

# ╔═╡ 9cb67fc0-5233-449f-8170-f82cb3ab6761
c = sin(cos(12))

# ╔═╡ 96024106-5097-4ea3-9fcb-8ad40764dd06
course_name = "Machine Learning"

# ╔═╡ 6bb480b0-5c27-4e0e-ad5e-500466f95bc8
μ = 3.6

# ╔═╡ d1529a24-2afc-46ef-98f1-02366667a330
ϕ = "Hello"

# ╔═╡ 294d801a-c457-42cd-ad14-aa4e3aea7b08
md"Non mathematical unicode characters are also avilable as variable names"

# ╔═╡ 5029d3a4-9c28-4d58-9b6d-a20a18e51f79
😄 = 12 # type \:smile: then hit <TAB>

# ╔═╡ e3eafada-b863-46ed-ad43-25d7a67cd177
md"Multiple variable names can be assigned on a single line"

# ╔═╡ 2743250b-bd70-4bfc-94c9-a7c4c23378e1
x,y,z = 1, 3, 5

# ╔═╡ b668a845-8b70-461b-b401-635990490705
x

# ╔═╡ 36cb1987-77cd-44c2-835a-fa3307d13d16
y

# ╔═╡ b1264db6-02ff-4e22-987d-831f7267915d
z

# ╔═╡ 9d51e87f-0fe9-4cb1-a5cd-56190ebf1287
md"Variable names consisting of only underscores are special class of identifiers in Julia as they can only be assigned values but cannot be used to assign values to other variables"

# ╔═╡ d6192ed4-179d-4e98-8db3-52e8b2b0b3fd
___ = "Cat"

# ╔═╡ f057295c-5f2a-4809-8302-100d8506c249
md"""
But the folowing assignment give an error
`mycat = ___`
"""

# ╔═╡ cc1226f4-514d-4bab-9a5f-2dae7335be3c
md"""

!!! tip 
	For performance reasons global variables should be defined as constants. Global variables are variables not defined in any local scope.

"""

# ╔═╡ 7c3116c8-8926-4fd6-acae-e98c17ffe94f
md"""
### Excercise

Which of the following are valid variable names in Julia.? Select as many answers that apply.
"""

# ╔═╡ 8511d2dc-5010-42bb-b80e-a8a1364c8fbb
@bind __varnames__ confirm(MultiCheckBox(["ben_01", "01_name", "initial_state", "hello.world", "Address", "_my_variable"], orientation = :column))

# ╔═╡ 3bd55c7b-d785-4d17-9705-3125dafd3a1a
md"""
## Arithmetic Operators
Basic arithmetic operators in Julia are very much similar to those in Python. These operators are either unary or binary. The binary operators support both infix and prefix notation.
"""

# ╔═╡ 232c90b7-55cf-4628-9c3e-e36ad475792a
2 + 2 #infix notation

# ╔═╡ a3ba45f5-2d35-4cf5-af75-ab5f4218c09d
+(2, 2) # prefix notation

# ╔═╡ 52bd1a35-8c31-4150-a2b2-07b88b9ffa54
md"""
The table below shows the full list of arithmetic operators supported in Julia

| Expression | Name           | Description                             |
|:---------- |:-------------- |:----------------------------------------|
| `+x`       | unary plus     | the identity operation                  |
| `-x`       | unary minus    | maps values to their additive inverses  |
| `x + y`    | binary plus    | performs addition                       |
| `x - y`    | binary minus   | performs subtraction                    |
| `x * y`    | times          | performs multiplication                 |
| `x / y`    | divide         | performs division                       |
| `x ÷ y`    | integer divide | x / y, truncated to an integer          |
| `x \ y`    | inverse divide | equivalent to `y / x`                   |
| `x ^ y`    | power          | raises `x` to the `y`th power           |
| `x % y`    | remainder      | equivalent to `rem(x,y)`                |

"""

# ╔═╡ 5c7cf4ef-83ec-4f76-9ac0-e6aed6211fe4
md"""
## Logical and Bitwise Operators
Basic Logical and  Bitwise operators in Julia are also similar to those in Python. Logical and Bitwise Operators also support both infix and prefix notation, with the exception of short-circuiting operators (`||` and `&&`) which only supports infix notation.
"""

# ╔═╡ c427167b-81c2-40e9-8cf4-1bf642b42af9
9 == 9

# ╔═╡ 1fa43f2a-787a-4bdf-b2ce-654279c18283
1 < 2

# ╔═╡ a2a297eb-d55e-4312-8ab4-2ccaba15b35a
2<3<=5 # Chaining

# ╔═╡ f4e49ea4-954b-4151-9232-c41cfcfd862e
true | false

# ╔═╡ 67698825-ff37-43e8-bb75-a2b768c2cf84
false || 5

# ╔═╡ c9a6bd71-4719-4d87-99ef-e9955092837a
sin(30) < 1 && 5

# ╔═╡ 7546e3df-acee-4392-9693-f4844873fd0d
md"""
The table below displays the list of Basic Boolean and Bitwise Operators in Julia

| Expression | Name                                                                     | Operator Type |
|:---------- |:------------------------------------------------------------------------ |:------------- | 
| `!x`       | negation                                                                 | Boolean
| `x && y`   | [short-circuiting and](@ref man-conditional-evaluation)                  | Boolean
| `x \|\| y` | [short-circuiting or](@ref man-conditional-evaluation)                   | Boolean
| `~x`       | bitwise not                                                              | Bitwise
| `x & y`    | bitwise and                                                              | Bitwise
| `x \| y`   | bitwise or                                                               | Bitwise
| `x ⊻ y`    | bitwise xor (exclusive or)                                               | Bitwise
| `x ⊼ y`    | bitwise nand (not and)                                                   | Bitwise
| `x ⊽ y`    | bitwise nor (not or)                                                     | Bitwise
| `x >>> y`  | [logical shift](https://en.wikipedia.org/wiki/Logical_shift) right       | Bitwise
| `x >> y`   | [arithmetic shift](https://en.wikipedia.org/wiki/Arithmetic_shift) right | Bitwise
| `x << y`   | logical/arithmetic shift left                                            | Bitwise
"""

# ╔═╡ 1ad26a93-a030-4c43-9223-1de901a0d7ba
md"""
## Control Flow
Julia provides several standard control flow mechanisms present in High-level programming languages. For more info see [Control Flow](https://docs.julialang.org/en/v1/manual/control-flow/) section of the Julia Documentation.

### Compound Expressions
```julia
begin
	#do_stuff
end
```
begin blocks do not introduce any new scope and are commonly used in **Pluto** notebooks for evaluating multi-line code in a single code cell.

"""

# ╔═╡ 4a510d19-92ee-4159-99b9-aed045ea7f23
var3 # no new scope is created

# ╔═╡ 539ae2d5-cac1-48f0-809f-863e6c03cd8f
md"""
### Conditionals
```julia
if test
	#do_stuff
end
```

```julia
if test
	#if branch
	# do_stuff
else
	#else branch
	# do_stuff
end
```

```julia
if test_1
	#branch 1
	#do_something
elseif test2
	#branch 2
	#do_something
	.
	.
	.
else
	#else brach
	#do_something
end
```
Conditional expressions are the most common control flow mechanism and are found in almost every program. Conditional expressions intoduce a new local scope, hence new variables defined in any branch in a conditional expression cannot be accessed in the parent scope.
"""

# ╔═╡ 30611c7c-e904-4e91-8fd1-1342c84fadd0
md"""
Below is what the body of absolute value function might look like, for a given input `number`. 

number =  $(@bind number Scrubbable(-10:10, default=5))
"""

# ╔═╡ 6b28703d-eed9-4a1c-a6fd-ef9e6fcc01d1
if number < 0
	-number
else
	number
end

# ╔═╡ efe29053-67ce-4ae1-9d9d-75757b96bb2b
 md"""
### Loops
Julia provides looping mechanism and syntax similar to those available in high-level programming languages. But unlike the loops in most high-level programming languages, loops in Julia are blazinly fast.
#### While Loop
```julia
while true
 #do_some_stuff
 if exit_condtion
 	break
 end
end
```

```julia
while true
 if exit_condtion
 	break
 end
 #do_some_stuff
end
```

```julia
while exit_condition
 	#do_stuff
end
```
While loops are the most basic loop structure in Julia. While loops introduces a new local scope in it's parent scope.
"""

# ╔═╡ 84dd7b92-022a-4c42-bcb5-5b0d5dfb4337
md"""
The code below uses a `while` loop to compute `exp(2)` with an approxiate relative error less than $$1\%$$
"""

# ╔═╡ 529c8653-bfe6-428d-8c9f-2cf6b60a993b
begin
	exp2old = 1
	fact = 1
	index = 1
	pow2 = 2
	reltol = 1
	while true
		exp2new = exp2old + pow2/fact
		index += 1
		fact *= index
		pow2 *= 2
		ϵₐₚₚᵣₒₓ = abs((exp2new - exp2old)/exp2old) * 100
		exp2old = exp2new
		ϵₐₚₚᵣₒₓ < reltol && break
	end
	exp2old
end

# ╔═╡ 619e95d3-ce04-4f12-99ed-6f4e0248df8a
md"""
#### For Loop
```julia
for i in iter
 	#do_some_stuff
end
```
```julia
for i in iter1, j in iter2, ...
 	#do_some_stuff
end
```

For loops are useful in repeating some operations a known number of times. For loops like while loops and conditionals introduce a new local scope.
"""

# ╔═╡ 6a532765-be71-44d0-8a70-2166b87fbc95
md"The following program shows how a for loop can be used to display the first ten positive integers"

# ╔═╡ 4d7b02cd-2560-4f4d-9854-266f12051afd
with_terminal() do 
	for i in 1:10
	@show i
	end
end

# ╔═╡ 6d6b610e-1ec1-49e5-bdca-574d9f12694d
md"The code below shows how a for loop can be used to display all the elements of the cartesian product, $$AXA$$, where $$A = \{1, 2, 3, 4\}$$"

# ╔═╡ 5bb4dae6-f0fb-4116-80e8-37a588427a5c
with_terminal() do 
	for i in 1:4, j in 1:4
		@show i,j
	end
end

# ╔═╡ b0996b6e-0223-4e34-9295-48d1810bd9e8
md"""
### Excercise

Using a while loop, compute the first number in the fibonaci sequence which is greater than 50 and assign the result to the variable `fib50`. 

Note: Take the first two fibonaci numbers as 0 and 1
"""

# ╔═╡ 51ad94ec-75ef-4d40-a152-13ec595d60cc
begin
	# input your code here
	a=1
	b=1
	while b<= 50
		temp=b
		b=a+b
		a=temp
	end
	fib50=b
end

# ╔═╡ c56e6776-0330-4077-ae96-841e1e5fe150
md"""
## Tuples and NamedTuples
These are built-in immutable data structures of known length, similar to Python's `tuple` and `namedtuple`. NamedTuples are sometimes treated as tabular data sources.
"""

# ╔═╡ 8712216b-d9e9-4240-bdd5-c60269bb1931
my_tuple = (1, 3, 5)

# ╔═╡ 42a8cfbd-1455-4a8c-b254-fc695a2fd15e
typeof(my_tuple)

# ╔═╡ 98a559b3-9c8f-4a62-aa0c-bcd7e9c8df7b
my_tuple[3]

# ╔═╡ 5d907921-8af0-493f-b392-c8dddc9108dc
length(my_tuple)

# ╔═╡ c14194cf-722d-40fd-880e-a1179c5fdbef
my_named_tuple = (names = ["John", "Jane", "Jade"], ages = [29, 23, 25])

# ╔═╡ 6c73c8e3-f323-4d88-9265-daa97fc7dd5a
typeof(my_named_tuple)

# ╔═╡ 6fdd92c8-c8e5-45f2-b499-7b7e381e908e
length(my_named_tuple)

# ╔═╡ 41ec7957-9dd6-4e62-8221-4e6bf8c0115d
my_named_tuple.ages

# ╔═╡ 8aaaba8b-ac76-4d39-b2f6-5bd9edaa5ff4
my_named_tuple.names

# ╔═╡ c9191d66-8e06-4213-b4a8-e3971e20bedb
md"""

!!! note 
	By default Julia uses 1-based indexing for indexing into Tuples and Arrays etc. This is similar to Matlab and Fortran Arrays which also  use 1-based indexing but may make Python and C programmers feel uneasy as they are used to 0-based indexing. We restrain ourselves from joining the war between 0-based indexing and 1-based indexing. 

"""

# ╔═╡ 06d62230-7598-4b3c-88a1-c4e55f942b99
md"""
### Excercise

Encode the dataset shown below as a Julia `NamedTuple` of vectors. Each entries in the table should be encoded as a string

| Country       | Capital          | HeadofState  |
| :------------ | :--------------- | :------------- |
| Brazil        | Brasília         | Dilma Rousseff |
| Cameroon      | Yaoundé          | Paul Biya      |
| China         |  Xi Jinping      | Beijing        |
| Germany       | Berlin           | Joachim Gauck  |
| United States | Washington, D.C  | Joe Biden      | 
"""

# ╔═╡ 1c2e1680-af72-49a0-968c-a32559ed7a20
countries = missing

# ╔═╡ 128ba956-de90-4efe-acc6-6dbfcd3ede57
md"""
## Functions

Every programming requires some means of creating abstractions. Functions are one way of creating abstractions in Julia. Functions are first class objects in Julia. Julia has several built-in functions defined across it's `Base` module and Standard Libraries (e.g `LinearAlgebra`, `Random`), and also provides programmers with syntax for defining new functions. The amazing thing is that user defined functions runs as fast as in-built functions, because most of Julia is written in Julia.

### Creating new functions
#### Basic functions
"""

# ╔═╡ 523530f9-e619-4ff5-a590-5e7813533cc1
expit(x) = 1/(1+exp(-x))

# ╔═╡ 2c23bcc3-e04f-4165-9b9a-175a2e661b59
expit(1.5)

# ╔═╡ 712d9257-ceb9-4e69-a7a7-371c1718d445
function is_pythagoras_triplet(x, y, z)
	# a, b, c are the sorted version of x, y and z
	# Non-mutating version of bubble sort is performed next
	a,b,c = x, y, z
	if b < a
		a , b = b, a
	end
	if c < b
		b, c = c, b
	end
	
	if a^2 + b^2 == c^2
		return true
	else
		return false
	end
end

# ╔═╡ 7d7e2c2e-799a-4801-886e-c1747663a7da
is_pythagoras_triplet(6,8,10)

# ╔═╡ 1c4c22d6-8199-4ac6-8034-3bf207ff31a4
mtuple= (6,8,10)

# ╔═╡ 16312da0-0d8f-4c1c-a59d-0e644f55ecba
is_pythagoras_triplet(mtuple...)

# ╔═╡ 79d83c21-d1b9-4ee6-922a-eb1e91ce8c1c
md"""
#### Functions with optional, keyword arguments and Variable arguments
Julia also allows for creation of functions with optional and/or keyword arguments.
```julia
function fname(arg1, arg2=3, args...; kwarg1, kwarg2=:a, kwargs...)
	#do_stuff
end
```

"""

# ╔═╡ 6f8a839a-2d68-413f-9d47-de6d9dedc9ef
mysum(a, b...) = reduce(+, (a, b...))

# ╔═╡ 44a88429-aa6b-43cf-8473-217a13edd568
mysum(2, 3, 4)

# ╔═╡ 4875e350-d409-4400-b29e-3a12f19c4d1c
function fib(n)
	a=1;b=1;
	for i=3:n
		temp=b
		b=a+b
		a=temp
	end
	return b
end

# ╔═╡ ae49a168-3ab5-41bd-b0fa-8e72b550d606
fib(40)

# ╔═╡ 4f4b3624-4ba6-442a-996e-d93cac846e47
function fib2(n)
	if n==1 || n==2 
		return 1
	end
	return fib2(n-1)+fib2(n-2)
end

# ╔═╡ ba491049-c8e1-49fb-87f6-bff6d5d3bae3
@time fib2(40)

# ╔═╡ abccc425-4ac5-4285-80e6-ac7b2390993c


# ╔═╡ cc7d6f03-045d-4a8a-a2e6-ec51bb0cc4a6
@memoize function fib3(n)
	if n==1 || n==2 
		return 1
	end
	return fib3(n-1)+fib3(n-2)
end

# ╔═╡ a864be88-ba2a-40a6-bcc0-e5147a79b668
@time fib3(40)

# ╔═╡ c33b7c24-283f-4859-97c0-d225c7971633
md"""
#### Functions Composition
Julia provides programmers with two ways of composing functions, `∘`(function composition operator) and `|>` (pipe operator).

"""

# ╔═╡ af2a3500-4120-476f-8b00-3ba3d0867977
(cos ∘ sin)(pi)

# ╔═╡ 4b15287b-3df2-4603-b909-37b6051bf76f
(cos ∘ sin)(pi) == cos(sin(pi))

# ╔═╡ 4c38e421-c314-4931-85de-2f6079955d7b
pi |> sin |> cos

# ╔═╡ c1483996-5ddc-400c-81ee-07249b5f4619
md"""
#### Annoynymous functions and Do Block syntax
An annoymous function is a function without any named assigned to it. Annoymous functions cannot be accessed after creation, hence they are usually passed directly to higher order functions(e.g `map`, `broadcast`, `open` etc.). There are two ways anonymous functions can be created in julia.

```julia
function (args)
	#do stuff
end
```
or 
```julia
(args)->(#do_stuff)
```

The Do-Block syntax provides a more convenient way for programmers to pass more complicated anonymous function to higher order functions with the first argument accepting a function as first argument.
"""

# ╔═╡ 06801aec-662e-450f-868b-12c7be4dd7b1
map((x)->(x^2 + 1), (1, 2, 3, 4, 5))

# ╔═╡ 9b565ae6-2af9-460f-b7d1-d0e999a37d0b
map((1, 2, 3, 4, 5)) do x
	x^2 + 1
end

# ╔═╡ fad73216-dbc6-4eb5-9d0b-1a37091af5fa
md"""
## Characters and Strings
Characters in Julia are represented via the 32-bit primitive type `Char`. The `Char` type can represent any `Unicode` character. Strings in Julia are finite sequences of characters. Strings are represented by the built-in type `String`. Both `Char` and `String` objects are immutable.

Aside the `Char` and `String` primitive types, Julia also has built-in support for non-standard string literals (e.g bytestring and regular expression) and programmers can even define their own character and string types for specialized cases.
"""

# ╔═╡ ff80c05d-f286-4c4c-9228-453034e2e823
md"""
### Characters

Character Literal
"""

# ╔═╡ 6e4f8b25-a012-4138-a505-1698d15ed348
mycharacter = 'o'

# ╔═╡ b80ff287-29b6-45e0-8910-3f36dbc1e94e
typeof(mycharacter)

# ╔═╡ 9038c6d8-fdf4-4228-8689-8000a0948b49
md"Construction from valid Unicode Points. Valid Unicode code points are U+0000 through U+D7FF and U+E000 through U+10FFFF"

# ╔═╡ a8fc1b81-648a-45c3-8207-9aa6452f34bd
'\Ud70'

# ╔═╡ ea699478-d4b9-4585-af2d-4d42ff59f63d
Char(0x0000)

# ╔═╡ 4431618d-9519-4ad4-adde-9891b5bea9e0
md"Conversion to Integers"

# ╔═╡ 08bd7450-c786-48e7-8bd7-4358a0505b65
Int('a')

# ╔═╡ 6d2e85c1-14dc-4698-b1f0-8d705d699e44
md"Character comparison"

# ╔═╡ 75dd21ea-d3ab-4e56-886f-581baad8ec88
'A' <= 'X' <= 'Z'

# ╔═╡ 547bbbf7-3d64-4277-83fc-a127c7f03c1e
md"""
### Strings

String Literal
"""

# ╔═╡ 2727be0b-5bee-4bd8-a744-07ab5901fe44
str = "Hello World"

# ╔═╡ 8eccda41-939e-45e3-9762-6298704d15c0
typeof(str)

# ╔═╡ 1e04b229-6be9-4708-aa8e-c46de78d1c9c
"""
This is another string literal represented with triple quotes. 
It can even embed "double quoted string" similar to Python
"""

# ╔═╡ 427234d0-be43-47c0-93e1-4e3c7af09e58
md"String Concatenation"

# ╔═╡ 6192af5b-e405-4532-ae43-200dc199de9d
"Hello" *" World!"

# ╔═╡ 2b65cc6a-5059-4cc8-ae91-867ab5ec7d2f
md"String Interpolation"

# ╔═╡ 62bdffa1-1fbd-46fb-9696-83ada184f77f
firstname = "John"

# ╔═╡ 5e70c188-04c7-4464-8ed2-f1c2c067abd0
lastname = "Joe"

# ╔═╡ 80eb9a45-ac2b-4243-8be6-d49bab32d079
"Hello $(firstname) $(lastname), Welcome to the course"

# ╔═╡ 4d8d55bf-c9e8-433d-90bf-6ef5b51cce7d
md"String Indexing"

# ╔═╡ 9dde4458-1044-4cdb-965d-32bbc39fdf8f
str[1]

# ╔═╡ 99aa4c2b-862a-4ba3-b8ff-418dda74b4fb
str[end]

# ╔═╡ b1433e12-6a9c-4324-9811-a3bd3e71ca0f
str[[1:2:5;]]

# ╔═╡ e24d5069-3053-4c33-8855-e218c9f0d85a
[1:2:5;]

# ╔═╡ d5ae4a57-cb05-417c-9a31-97e77d657fbe
md"Lexographic Comparison of Strings"

# ╔═╡ 27fe7d76-9525-41b6-b2e6-6e991e3abdaa
"abc" < "def"

# ╔═╡ 442723b0-36a3-4855-bf09-5799c78cf716
"ring" < "ringworm"

# ╔═╡ fe4fa23e-945a-4264-bec9-bb09a450616d
"ring" == "ringworm"

# ╔═╡ 465625c7-d0b9-4f6a-91e8-4ec092d891de
md"""
Common built-in string functions include
- `occursin`
- `findfirst`
- `findnext`
- `thisind`
- `prevind`

To see documentation for any of these functions you may use Pluto's Live docs option.
"""

# ╔═╡ c67d5547-b8fd-46e3-8ca8-47e851da1428


# ╔═╡ e11a55d2-f14a-464e-9c2e-67a2fef30e89
md"""
### Excercise
**1.** Write a function that accepts an input string, `str` and returns a string containing only the vowels in `str` subject to the following specifications.

a. If there are no vowels in the input string, your function must return the empty string literal i.e ("").

b. The characters in vowel string obtained from your function appear in the same order as in the input string. 

c. If a the input string conatins a repeated vowel character, then the output string should also include the vowel character repeated the same number of times as in the input string. For example if the input string is "taremamasalta", then the output string should be "aeaaaa".  
"""

# ╔═╡ b233ff17-855d-4ac9-b06d-8fe483bb4d63
function vowel_string(str)
	return missing
end

# ╔═╡ 0a260c90-476d-4190-be50-7019a01c317a
md"""
**2.** Using the `vowel_string` function you just defined above, get the vowel string of the string "MachineLearning". 
"""

# ╔═╡ b7a92f45-195a-4301-9ea5-3bfbdfb351ac
vstring = vowel_string("MachineLearning") #missing

# ╔═╡ 8b0f8b43-7233-422b-a579-6a4e8df1c77a
md"""
## Arrays

Julia like Matlab, Python and R has built-in support for multi-dimensional Arrays. Arrays are stored in column-major similar to Fortran, R and Matlab Arrays but unlike C arrays which are row-major. Unlike in most other programming Languages where arrays are cosidered mutable data structures, in Julia not all arrays are mutable e.g `UnitRange`. The Language also allows programmers to define specialized arrays objects. See [Interfaces Section of Julia Documentation](https://docs.julialang.org/en/v1/manual/interfaces/#Interfaces)

### 1-D Arrays
1-dimensional arrays in julia are instances of the `AbstractVector` type. 
"""

# ╔═╡ ea77cbb4-11fc-460f-9593-c12ecb06ba45
my_array = [1, 2, 3]

# ╔═╡ 5d395d27-221a-48a6-9567-70d7cf636705
typeof(my_array)

# ╔═╡ cf7caf2c-146d-424f-beb6-a2a8cef12321
isa(my_array, AbstractVector)

# ╔═╡ 2ebcf881-35dc-4f03-98bf-39b014535805
my_range = 0:2:20

# ╔═╡ 2659f1db-9a27-4bd0-a471-0c38824e659f
isa(my_range, AbstractVector)

# ╔═╡ e495d916-dcb2-471c-9c7c-dc61aecb580e
my_range == [my_range;]

# ╔═╡ aed984b7-dd7b-474b-bb78-ee0397e681b3
eltype(my_array), eltype(my_range), length(my_array), length(my_range)

# ╔═╡ 9eab56e3-68e5-40df-b403-b66da5e71b17
md"Vertical Concatenation"

# ╔═╡ 019912a7-fdcb-4776-bf0f-913dca46dafd
[my_range; my_array]

# ╔═╡ b5eb8829-cb85-400f-ba7d-5b87521d4ba6
vcat(my_range, my_array)

# ╔═╡ ccb1dcdc-522e-4c0d-be2c-a3f1eb19348c
md"1-dimensional Array Indexing"

# ╔═╡ 97103d90-8327-452a-be66-3be646968689
my_range[[1, 3, 4, 8]]

# ╔═╡ 7dd8344d-4f7d-4a84-83f8-3a44a1cc88e7
my_range[[1 3; 4 8]]

# ╔═╡ d0bac7ad-d90f-4001-917c-27605c365c03
begin
	my_array3 = [1.0; 3; 6] #yet another way of constructing 1-d Array
	my_array3[2] = 0
	my_array3
end

# ╔═╡ 0882a6f7-f1d4-4f72-a1b2-a43524ab715c
md"Iterating 1-D arrays"

# ╔═╡ 40cddda3-acbb-47e9-ab57-939d3c12a6c6
with_terminal()do
	for i in my_array3
		@show i
	end
end

# ╔═╡ 07a07350-0156-415b-bfdc-ca55da2cd68b
methods(show)

# ╔═╡ fa2bc10b-8d0b-4993-86cc-4e7cec9a44a6
with_terminal()do
	@inbounds for i in eachindex(my_array3)
		println("my_array3[$i] = $(my_array3[i])")
	end
end

# ╔═╡ 701235ab-0d24-4e1c-952d-bbb8f3f61be2
even_integers = [i for i in 1:10 if i%2==0]

# ╔═╡ 1edb759a-48a1-4e98-bc38-563c8d1680a7
md"Logical indexing (Masking) is also suppoted as in Python Numpy Arrays"

# ╔═╡ cabd0588-e36f-4fe4-9db3-0403d25c656a
even_integers[even_integers .< 5]

# ╔═╡ 84be58fe-41ab-4d1a-8242-03ad68e6a952
md"""
### 2-D Arrays
2-dimensional arrays in julia are instances of the `AbstractMatrix` type.
"""

# ╔═╡ 8f65f71b-3fd1-45a1-92e1-93baae1e8e4d
my_2d_array = [1 4 7; 2 5 8; 3 6 9]

# ╔═╡ b8a550bf-8381-44ba-be92-5805d84b7f98
isa(my_2d_array, AbstractMatrix)

# ╔═╡ 513c10a1-31f0-4c7d-8102-1cbab98a84ea
md"Construction from 1-d arrays"

# ╔═╡ aae3ac16-1c4b-4a7b-8039-713d98fd9ea5
my_2d_array2 = reshape(1:9, 3, 3)

# ╔═╡ a215bf4e-8d96-4cbb-8407-29b2907f3d5c
isa(my_2d_array2, AbstractMatrix)

# ╔═╡ 85ca81bc-030c-4ce2-a486-e38923f85368
md"Vertical Concatenation"

# ╔═╡ ce3d30d6-d8b7-4ae4-86f0-3a2fce3a1bca
[my_2d_array; [1 1 1]]

# ╔═╡ 33ca2228-7962-4818-908a-12a346cdbc86
vcat(my_2d_array, [1 1 1])

# ╔═╡ 99b8bd9e-0045-485d-a53a-b021199536ea
md"Horizontal Concatenation"

# ╔═╡ 2a319b11-d7e5-4b50-a689-46122514bc81
[my_2d_array, [1, 1, 1]]

# ╔═╡ 5ec7d923-b2cd-4690-bd14-448a1c73d12c
hcat(my_2d_array, [1,1,1])

# ╔═╡ 23e10e87-b78c-422f-8c31-a5ab0647a13d
md"2-dimensional Array indexing"

# ╔═╡ 4dca08e3-0cfc-4b2c-8b05-a4bf4ada3458
my_2d_array[1,2]

# ╔═╡ 2610b1f7-180a-4732-a0dd-597a0685ddda
my_2d_array[3]

# ╔═╡ 42b5c883-4849-45fc-8a81-fe48dcbaeffc
my_2d_array[[1, 3], [2, 3]]

# ╔═╡ 2468dc1f-8721-403e-b195-cdd3d0da3558
begin
	my_2d_array3 = Float64[9 3 7; 2 3 4]
	my_2d_array3[2, 3] = 0
	my_2d_array3
end

# ╔═╡ 14a0eae8-be5d-4652-b187-6c3cfd629c62
md"2-d Arrays also support logical indexing"

# ╔═╡ 6d93ac7a-3634-4ab8-8aaa-311e5a2d657a
my_2d_array[my_2d_array .> 2]

# ╔═╡ 1b59151d-6b90-4df4-bc96-3997272cd8d7
md"""
Iterating 2-D arrays
"""

# ╔═╡ 0c4066c0-5ee7-4029-9e88-3b96bde0b19d
with_terminal()do
	@inbounds for ind in eachindex(my_2d_array)
		println("my_2d_array[$ind] = $(my_2d_array[ind])")
	end
end

# ╔═╡ 8e110723-c9fe-4a65-8a9c-1ed7b2b1c5f6
with_terminal()do
	@inbounds for j in axes(my_2d_array, 2), i in axes(my_2d_array, 1)
		println("my_2d_array[$i, $j] = $(my_2d_array[i, j])")
	end
end

# ╔═╡ 211d4a37-ba55-4514-8cf0-27e665050c71
md"""
!!! info
	For more detailed info about 1-D, 2-D and Multi-Dimensional Arrays see [Muli-Dimensional Arrays](https://docs.julialang.org/en/v1/manual/arrays/).
	To see a wide range of useful custom array objects, checkout [Julia Arrays](https://github.com/JuliaArrays) github organization.
"""

# ╔═╡ 1dcec520-4e80-4403-ae2a-5e096c39d7f4
md"""
### Array broadcasting, Map and Reduce

Julia has several built-in higher order functions that allow programmers to apply a function element-wise to one or more collections. The `map` are `broadcast` functions are two common ones. Julia also has built-in higher order functions for applying a reducing function on a collection. The `reduce` function is one of such function. Unlike the `map` function the `broadcast` function is under certain conditions is capable of streching the smaller collection to match dimensions of the larger collection without copying. Let's look at some examples

"""

# ╔═╡ e6c6c0c2-d4c6-43d9-a57b-99f0faa8dd0f
map(sqrt, [1, 2, 3, 4])

# ╔═╡ da4585c8-254a-444e-83ea-d8a205840055
broadcast(cbrt, [1, 2, 3, 4])

# ╔═╡ 4e46f10a-69f5-4ee3-a7f9-7fe1f13e7391
map(+, [1 2; 3 4], [1, 1])

# ╔═╡ b62b8ca5-e746-43b7-a1ae-fd5f28a83f28
broadcast(+, [1 2; 3 4], [1, 1])

# ╔═╡ 28d0687f-97ef-44df-beb4-aa54af665482
md"Julia has syntatic sugar for the `broadcast` function as dot call syntax described below"

# ╔═╡ e6ab08be-4977-409b-a33a-382e8fcdb244
(+).([1 2; 3 4], [1, 1])

# ╔═╡ 2f43036d-d3cd-467b-bb27-5616cad0ca52
md"Since the `+` function is also an operator that supports infix form, the `broadcast` call could also be expressed as"

# ╔═╡ 0592f7c7-4239-4bb3-9c0b-a5779746d7df
[1 2; 3 4] .+ [1, 1]

# ╔═╡ 78d56dba-2c4a-41a5-acc7-7a3f48c6b207
reduce(vcat, [[1, 2, 3], [4, 5, 6]], init=[])

# ╔═╡ 70b093de-f991-49fe-a909-8082a6cf9bc0
reduce(+, [1, 2, 3], init=0)

# ╔═╡ 5c9830e2-b789-425f-8f1d-0952fab05c8a
md"""
!!! note
	The associativity of the `reduce` function is implementation dependent. This means that you can't use non-associative operations like - because it is undefined whether `reduce(-,[1,2,3])` should be evaluated as `(1-2)-3` or `1-(2-3)`. Use `foldl` or `foldr` instead for guaranteed left or right associativity.
"""

# ╔═╡ f2ba7898-d589-4112-87b0-7fc8817a6b12
md"""
### Excercise

Using either `broadcast` or `map`, apply the the `expit` function defined earlier, element-wise on the array, `arr` defined below.
"""

# ╔═╡ 66860084-849d-48ce-8273-fc10d31612fb
begin
	arr = [3.0, 2.2, 1.7, 0.6, 0.4, 4.3]
	expit_of_arr = missing
end

# ╔═╡ 04ab9804-2437-4696-bb72-ecaee95213a9
md"""
## Dictionaries
Dictionaries in Julia behave similarly to dictionaries in Python. The standard dictionary object in Julia is the `Dict`. Similar to `NamedTuples`, a `Dict` object may also used to encode tabular dataset, but unlike `NamedTuples`, `Dict` objects are mutable.

Dictionary Constructors
"""

# ╔═╡ 2480e4cd-393f-48d8-8ec7-22d8e381c560
mydict = Dict("d"=>4, "j"=>10, "z"=>26)

# ╔═╡ 25b80a64-d9e1-43a6-916b-e43755c252f8
typeof(mydict)

# ╔═╡ 3a387f24-7ee6-405c-8c06-d6efe2207a6f
mydict1 = Dict(i=>"a$i" for i in 1:3)

# ╔═╡ 037baedd-0fae-4098-afaf-e516268d7b81
mydict2 = Dict((1, 2)=>:egg, (2, 1)=>:tomatoes, (1, 1)=>:fish)

# ╔═╡ fa1230b0-8919-4f15-86a9-8b19627f3809
md"""
Dictionary indexing
"""

# ╔═╡ 40865cf5-6d91-4d15-90a7-738f8d3ce18f
mydict["j"]

# ╔═╡ 3c1f1fd8-9db0-4fdc-b90b-37e81319e606
mydict2[(1,1)]

# ╔═╡ 3804859c-ab83-4929-9b8b-b34961838d1f
md"Dictionary mutation"

# ╔═╡ 03a4a6bc-f37c-4a2b-ba78-5661ce6519b5
begin
	mydict3 = Dict([("d", 4), ("j", 10), ("z", 26)])
	mydict3["j"] = 11 #modifying values
	mydict3
end

# ╔═╡ 9bee7432-3ad9-4d01-9874-ef545cd5351d
begin
	mydict4 = Dict([("d", 4), ("j", 10), ("z", 26)])
	mydict4["a"] = 1 #adding new key `"a"`
	mydict4
end

# ╔═╡ bc9f297d-3035-4b7c-90d7-37773103f61f
begin
	mydict5 = Dict([("d", 4), ("j", 10), ("z", 26)])
	delete!(mydict5, "z") #remove the key-value pair `"z"=>26` from `mydict5`
	mydict5
end

# ╔═╡ a5d138b8-ec35-4734-89de-5743be03cc4c
md"Iterating Dictionaries"

# ╔═╡ 79921d4e-af5e-448f-a42e-f63a4a65dd83
with_terminal()do
	for (k,v) in mydict
		@show (k, v)
	end
end

# ╔═╡ e4f5223a-655b-4c44-91ed-955cfd43a2d8
with_terminal()do
	for k in keys(mydict)
		@show k
	end
end

# ╔═╡ f3ed4faa-aa3b-41fa-b755-b14c745c2692
with_terminal()do
	for v in values(mydict)
		@show v
	end
end

# ╔═╡ 99e69935-02dc-4daf-9576-ffc73dad4a2a
md"Conversion to Arrays"

# ╔═╡ 8c5a123f-dc5a-40e1-aa91-c8bb77e61527
collect(mydict)

# ╔═╡ 588c50a2-2a1b-4362-804a-9773e880f537
md"""
### Excercise
**1.** Write a function that accepts an input string, `str` and returns a dictionary with keys being the unique characters and values corresponding to the frequency of occurence of each unique character. 
"""

# ╔═╡ 8d73e153-34fd-439c-84fe-5438ef589790
function word_to_freq(str)
	return missing
end

# ╔═╡ d3e5df42-49d7-4c3b-b00f-ae296a104258
md"""
**2.** Using the function you've defined above, get the frequency analysis of the string "aghyaT" as a dictionary. 
"""

# ╔═╡ 6b59d6b1-fdbe-496f-9083-55da8d62c393
freq_dict = missing

# ╔═╡ 9dd546c7-0e84-4319-86cd-a4cb0e6291be
md"""
## Distributions and Random Numbers.

## Random Number Generaration 
Julia comes built-in with functionality for generating random numbers. This is exposed to the programmer via the `Random` standard library. The `Random` standard library consists of implemented Random Number Generators(`MerseneTwister`, `RandomDevice` `Xoshiro`) and random generating functions(`rand`, `randn`, `randexp`, `bitrand`, `shufle`, `randstring`). The library also provides an API that allows programmers to create new Random Generators and extend the random generating functions.
"""

# ╔═╡ 442278ec-e29f-4c83-bb8a-ac5d27b28e95
rand() #random numer from the uniform distribution [0,1]

# ╔═╡ 3ad081a4-a133-47df-8b0a-20173ff33d84
randn() #random numer from the standard normal distribution

# ╔═╡ 0a3b57b9-98fc-41d1-8238-f8079df849c6
randstring()

# ╔═╡ e7b131a1-948e-4b7b-853b-bbcbb53f2ec9
shuffle(1:10)

# ╔═╡ bd271f8b-0b47-4010-9606-d1a37318a7b5
randperm(10)

# ╔═╡ 0d9ce8a8-fa5b-43f5-8a32-eac088b2b428
md"""
## Distributions 
The [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) package provides programmers with a wide variety of implemented probability distributions and associated methods(`mean`, `var`, `pdf`, etc.)
"""

# ╔═╡ b881ed38-7f06-4561-86fd-1f6d1d9362cf
cdf(Normal(0, 1), 0)

# ╔═╡ b83c2287-14ec-4892-bf0f-c5898f571268
mean(Normal(0, 1))

# ╔═╡ fbfb3fd8-8b45-44a7-a2bd-aedc8b823331
var(Normal(0,1))

# ╔═╡ fbdeb6fb-4e1c-482c-871e-f2fccdf96687
md"The linear algebra routines are optimized for different special matrices/structures"

# ╔═╡ 94f7527d-0546-4713-aeb7-71dc66f1c17f
mat = Float64[1 2 3;4 5 6; 7 8 9]

# ╔═╡ 732f6bd9-da0f-414a-aec1-2bb225c302b9
dmat = Diagonal(mat)

# ╔═╡ dc67c9f3-5553-4cf6-8233-070bd04d3c10
inv(dmat)

# ╔═╡ 323f1256-bcde-4b07-a374-04e075980dae
eigvals(dmat)

# ╔═╡ 19e533c9-7bf7-4710-ae27-ba419c65f368
eigvecs(dmat)

# ╔═╡ f2a668a5-5f1e-41eb-9cf0-212960c3da49
md"The Linear algebra routines can also be applied to non-special matrices"

# ╔═╡ d5c95557-9efa-43ef-a6c6-12092109cb68
svd(mat)

# ╔═╡ 0d635435-7350-4afc-a061-7857d106c989
md"""
## Julia Resources
- [Julia Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/#Style-Guide)
- [Fast Track to Julia 1.0](https://juliadocs.github.io/Julia-Cheat-Sheet/)
- [Julia Documentation](https://docs.julialang.org/en/v1/)
- [Quant Econs Julia Cheat Sheet](https://cheatsheets.quantecon.org/julia-cheatsheet.html)
- [Matlab-Python-Julia Syntax differences](https://cheatsheets.quantecon.org/)
"""

# ╔═╡ 7052fa4f-cb75-4962-ab98-4f1c025f2355
__lecture__(text, lecture=true) = lecture ? text : md""

# ╔═╡ bc90ea38-0853-40b5-bae8-31186da91dce
__lecture__(md"""
## Linear Algebra
Julia exposes implementations of special/structured matrices(e.g `Hermitian`, `Symmetric`, `Tridiagonal`, etc.) and linear algebra routines(e.g `eigen`, `det`, `svd`, etc.) via it's `LinearAlgebra` standard library. These linear algebra routines are based off from the [Laplack Library](http://www.netlib.org/lapack/). For more information see [Linear Algebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/).
""")

# ╔═╡ 78ef14fa-7a3f-49a1-b5c9-3dd8d7d30881
__correct__(text=md"Great! You got the right answer! Let's move on to the next section.") = Markdown.MD(Markdown.Admonition("correct", "Got it!", [text]));

# ╔═╡ 8d69cfbb-94bc-4747-87f8-647c53f9ae6d
__keep_working__(text=md"The answer is not quite right.") = Markdown.MD(Markdown.Admonition("danger", "Keep working on it!", [text]));

# ╔═╡ 67bbfb34-3a18-4db4-8b7d-d2180cff2de1
if !isempty(__varnames__)
	local score = 0
	local incorrect = String[]
	for vname in __varnames__
		if vname in ["ben_01", "initial_state", "Address", "_my_variable"]
			score += 1
		end
		if vname in ["01_name", "hello.world"]
			push!(incorrect, vname)
		end
	end

	# logic for generating text for incorrect answers
	incorrect_text = if length(incorrect) == 1
				"\n" * incorrect[1] * " is not a valid variable name"
	else
		string(incorrect) * " are not valid variable names"
	end
	
	if score == 4
		if isempty(incorrect)
			__correct__(md"You selected all 4 correct answers")
		else
			__keep_working__(
			md"""
			Although you have selected all 4 correct answers, you have also selected some wrong answers.
			$(incorrect_text)
			""")
		end
	else
		__keep_working__(
			md"""
			You selcted only $(score)/4 correct answers.
			$(incorrect_text)
			""")
	end
end

# ╔═╡ 31eb9c61-d4d3-4dd9-9cba-b6aa0d9adc11
if ismissing(fib50)
	__keep_working__(md"replace missing with your answer")
elseif fib50 == 55
	__correct__()
else 
	__keep_working__(md"Fix the body of your while loop")
end
	

# ╔═╡ 86347366-68aa-4de4-8400-728a44726d6d
if ismissing(countries)
	__keep_working__(md"replace missing with your code")
	
elseif !isa(countries, NamedTuple{names, T} where {names, T <: NTuple{N, AbstractVector{S} where S} where N}) || isempty(countries)
		__keep_working__(md"`countries` must be a `NamedTuple` of Vectors")
elseif !(all(in((:Country, :Capital, :HeadofState)), propertynames(countries)) && length(keys(countries)) == 3) 
	__keep_working__(md"One or more keys of `countries` named tuple is invalid")
elseif !(length(countries.Country) == length(countries.Capital) == 				length(countries.HeadofState) == 5)
	__keep_working__(md"One or more entries of `countries` named tuple are absent")
else
	__countries_iter__ = zip(
		("Brazil", "Cameroon", "China", "Germany", "United States"),
		("Brasilia", "Yaoundé", "Xi Jinping", "Berlin", "Washington, D.C"),
		("Dilma Rousseff", "Paul Biya", "Beijing", "Joachim Gauck", "Joe Biden")
	)
	local _md__output = md""
	for iter in zip(countries.Country, countries.Capital, countries.HeadofState)
		if !(iter in  __countries_iter__)
			_md__output = __keep_working__(
				md"The observation $(iter) is invalid."
			)
		end
		break
	end
	isempty(_md__output) ? __correct__() : _md__output
end

# ╔═╡ 73483cf3-3e48-4ca9-9b37-e4433ef01799
if ismissing(vowel_string("abc"))
	__keep_working__(md"replace missing with your code")
else
	if vowel_string("abcdef") isa String
		if vowel_string("learning") == "eai" 
			if vowel_string("LEarnIng") == "EaI" 
				if vowel_string("+3typ,'") == "" 
					if vowel_string("MachineLearning") == "aieeai" 
						__correct__()
					else
						__keep_working__(
							md"""
							Something is wrong with the body of your code. Your code should preserve vowel multiplicity.
							"""
						)
					end
				else
					__keep_working__(
						md"""
						Something is wrong with the body of your code. Your code should return an empty string when the input string conatins no vowels.
						"""
					)
				end
			else
				__keep_working__(
					md"""
					Something is wrong with the body of your code. Your code doesn't consider strings conating Capital Letter Vowels.
					"""
				)
			end
		else
			__keep_working__(
				md"""
				Something is wrong with the body of your code. The ordering must of the vowels must be preserved.
				"""
			)
		end
	else
		__keep_working__(md"Something is wrong with the body of your code. Your function must return a value of type String")
	end
end

# ╔═╡ 35cb458b-56cd-426b-acd4-4932c380498e
if ismissing(vstring)
	__keep_working__(md"replace missing with your code")
elseif vstring == "aieeai"
	__correct__()
else
	__keep_working__(md"Check your code and try again")
end

# ╔═╡ e2033d92-cf0c-4963-a44e-38a2ed8ddb10
if ismissing(expit_of_arr)
	#Markdown.MD(Markdown.Admonition("danger", "Keep working on it!", [md"replace missing with your code"]));
	__keep_working__(md"replace missing with your code")
elseif expit_of_arr  == map(expit, arr)
	__correct__()
elseif expit_of_arr isa Array
	__keep_working__(md"Make sure you make use of either `map` or `broadcast` in your solution.")
else
	__keep_working__(md"Your answer must be of type `Array`. Make sure you make use of either `map` or `broadcast` in your solution.")
end

# ╔═╡ 136e37ca-52b9-4158-bf67-ea577b463754
if ismissing(word_to_freq("abc"))
	__keep_working__(md"replace missing with your code")
else
	if word_to_freq("abc") isa Dict{Char, T} where {T<: Integer}
		if word_to_freq("abaueaett") == Dict('a' => 3, 'b'=>1, 'u' => 1, 'e' => 2, 't' => 2)
			__correct__()
		else
			__keep_working__()
		end
	else
		__keep_working__(md"Your function must output a dictionary with `Char` keys and Integer values")
	end
end

# ╔═╡ 4321c5ae-23bb-47d5-b83d-1dbaced64edc
if ismissing(freq_dict)
	__keep_working__(md"replace missing with your code")
elseif freq_dict == Dict('g' => 1, 'a'=>2, 'h' => 1, 'y' => 1, 'T' => 1)
	__correct__()
else
	__keep_working__(md"Check your code and try again")
end

# ╔═╡ a34656db-ffd3-44fc-961e-98c0a9315214
var2=4

# ╔═╡ bac96fd7-461f-45ba-9c3d-ab233301e6be
begin
	var1 = cospi(30)
	var2 = exp(20)
	var3 = var1^2 + sqrt(var2)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Memoize = "c03570c3-d221-55d1-a50c-7939bbd78826"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Distributions = "~0.25.49"
Memoize = "~0.4.4"
PlutoUI = "~0.7.37"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c9a6160317d1abe9c44b3beb367fd448117679ca"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.13.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "96b0bc6c52df76506efc8a441c6cf1adcb1babc4"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.42.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "9d3c0c762d4666db9187f363a76b47f7346e673b"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.49"

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
git-tree-sha1 = "90b158083179a6ccbce2c7eb1446d5bf9d7ae571"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.7"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "0dbc5b9683245f905993b51d2814202d75b34f1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.1"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "SpecialFunctions", "Test"]
git-tree-sha1 = "65e4589030ef3c44d3b90bdc5aac462b4bb05567"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.8"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

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

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "56ad13e26b7093472eba53b418eba15ad830d6b5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.9"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Memoize]]
deps = ["MacroTools"]
git-tree-sha1 = "2b1dfcba103de714d31c033b5dacc2e4a12c7caa"
uuid = "c03570c3-d221-55d1-a50c-7939bbd78826"
version = "0.4.4"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e8185b83b9fc56eb6456200e873ce598ebc7f262"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.7"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "85b5da0fa43588c75bb1ff986493443f821c70b7"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "bf0a1121af131d9974241ba53f601211e9303a9e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.37"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "d3538e7f8a790dc8903519090857ef8e1283eecd"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.5"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

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

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

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

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

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
git-tree-sha1 = "5ba658aeecaaf96923dce0da9e703bd1fe7666f9"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c3d8ba7f3fa0625b062b82853a7d5229cb728b6b"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.1"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "25405d7016a47cf2bd6cd91e66f4de437fd54a07"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.16"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─abd26020-9fb9-11ec-162c-571e629a4da6
# ╟─3ed99cc3-cf30-48df-9515-f9359d9c2c92
# ╟─da931c4d-149a-4bb1-82b3-7a7ae3852f32
# ╟─fea89255-14da-4733-abc4-f34f9518440d
# ╟─20822d76-6e88-4051-b0c9-eaca30d50a96
# ╠═3c439fc7-29cc-45db-8a7c-15ccc1139a31
# ╠═14753e06-0cdb-4314-8ef3-c04d29ca3495
# ╠═a34656db-ffd3-44fc-961e-98c0a9315214
# ╠═686a0c83-842e-4c9b-a672-bb03ba2599e6
# ╠═66a7b5f6-d38b-4074-88db-32cfb57d2e27
# ╠═117cbe3b-b417-4560-ba57-f03c5f2db884
# ╟─23ddab9f-d996-4ecf-860a-7cc5412b226c
# ╠═9cb67fc0-5233-449f-8170-f82cb3ab6761
# ╠═96024106-5097-4ea3-9fcb-8ad40764dd06
# ╠═6bb480b0-5c27-4e0e-ad5e-500466f95bc8
# ╠═d1529a24-2afc-46ef-98f1-02366667a330
# ╟─294d801a-c457-42cd-ad14-aa4e3aea7b08
# ╠═5029d3a4-9c28-4d58-9b6d-a20a18e51f79
# ╟─e3eafada-b863-46ed-ad43-25d7a67cd177
# ╠═2743250b-bd70-4bfc-94c9-a7c4c23378e1
# ╠═b668a845-8b70-461b-b401-635990490705
# ╠═36cb1987-77cd-44c2-835a-fa3307d13d16
# ╠═b1264db6-02ff-4e22-987d-831f7267915d
# ╟─9d51e87f-0fe9-4cb1-a5cd-56190ebf1287
# ╠═d6192ed4-179d-4e98-8db3-52e8b2b0b3fd
# ╟─f057295c-5f2a-4809-8302-100d8506c249
# ╟─cc1226f4-514d-4bab-9a5f-2dae7335be3c
# ╟─7c3116c8-8926-4fd6-acae-e98c17ffe94f
# ╟─8511d2dc-5010-42bb-b80e-a8a1364c8fbb
# ╟─67bbfb34-3a18-4db4-8b7d-d2180cff2de1
# ╟─3bd55c7b-d785-4d17-9705-3125dafd3a1a
# ╠═232c90b7-55cf-4628-9c3e-e36ad475792a
# ╠═a3ba45f5-2d35-4cf5-af75-ab5f4218c09d
# ╟─52bd1a35-8c31-4150-a2b2-07b88b9ffa54
# ╟─5c7cf4ef-83ec-4f76-9ac0-e6aed6211fe4
# ╠═c427167b-81c2-40e9-8cf4-1bf642b42af9
# ╠═1fa43f2a-787a-4bdf-b2ce-654279c18283
# ╠═a2a297eb-d55e-4312-8ab4-2ccaba15b35a
# ╠═f4e49ea4-954b-4151-9232-c41cfcfd862e
# ╠═67698825-ff37-43e8-bb75-a2b768c2cf84
# ╠═c9a6bd71-4719-4d87-99ef-e9955092837a
# ╟─7546e3df-acee-4392-9693-f4844873fd0d
# ╟─1ad26a93-a030-4c43-9223-1de901a0d7ba
# ╠═bac96fd7-461f-45ba-9c3d-ab233301e6be
# ╠═4a510d19-92ee-4159-99b9-aed045ea7f23
# ╟─539ae2d5-cac1-48f0-809f-863e6c03cd8f
# ╟─30611c7c-e904-4e91-8fd1-1342c84fadd0
# ╠═6b28703d-eed9-4a1c-a6fd-ef9e6fcc01d1
# ╟─efe29053-67ce-4ae1-9d9d-75757b96bb2b
# ╟─84dd7b92-022a-4c42-bcb5-5b0d5dfb4337
# ╠═529c8653-bfe6-428d-8c9f-2cf6b60a993b
# ╟─619e95d3-ce04-4f12-99ed-6f4e0248df8a
# ╟─6a532765-be71-44d0-8a70-2166b87fbc95
# ╠═4d7b02cd-2560-4f4d-9854-266f12051afd
# ╟─6d6b610e-1ec1-49e5-bdca-574d9f12694d
# ╠═5bb4dae6-f0fb-4116-80e8-37a588427a5c
# ╟─b0996b6e-0223-4e34-9295-48d1810bd9e8
# ╠═51ad94ec-75ef-4d40-a152-13ec595d60cc
# ╟─31eb9c61-d4d3-4dd9-9cba-b6aa0d9adc11
# ╟─c56e6776-0330-4077-ae96-841e1e5fe150
# ╠═8712216b-d9e9-4240-bdd5-c60269bb1931
# ╠═42a8cfbd-1455-4a8c-b254-fc695a2fd15e
# ╠═98a559b3-9c8f-4a62-aa0c-bcd7e9c8df7b
# ╠═5d907921-8af0-493f-b392-c8dddc9108dc
# ╠═c14194cf-722d-40fd-880e-a1179c5fdbef
# ╠═6c73c8e3-f323-4d88-9265-daa97fc7dd5a
# ╠═6fdd92c8-c8e5-45f2-b499-7b7e381e908e
# ╠═41ec7957-9dd6-4e62-8221-4e6bf8c0115d
# ╠═8aaaba8b-ac76-4d39-b2f6-5bd9edaa5ff4
# ╟─c9191d66-8e06-4213-b4a8-e3971e20bedb
# ╟─06d62230-7598-4b3c-88a1-c4e55f942b99
# ╠═1c2e1680-af72-49a0-968c-a32559ed7a20
# ╟─86347366-68aa-4de4-8400-728a44726d6d
# ╟─128ba956-de90-4efe-acc6-6dbfcd3ede57
# ╠═523530f9-e619-4ff5-a590-5e7813533cc1
# ╠═2c23bcc3-e04f-4165-9b9a-175a2e661b59
# ╠═712d9257-ceb9-4e69-a7a7-371c1718d445
# ╠═7d7e2c2e-799a-4801-886e-c1747663a7da
# ╠═1c4c22d6-8199-4ac6-8034-3bf207ff31a4
# ╠═16312da0-0d8f-4c1c-a59d-0e644f55ecba
# ╟─79d83c21-d1b9-4ee6-922a-eb1e91ce8c1c
# ╠═6f8a839a-2d68-413f-9d47-de6d9dedc9ef
# ╠═44a88429-aa6b-43cf-8473-217a13edd568
# ╠═4875e350-d409-4400-b29e-3a12f19c4d1c
# ╠═ae49a168-3ab5-41bd-b0fa-8e72b550d606
# ╠═4f4b3624-4ba6-442a-996e-d93cac846e47
# ╠═ba491049-c8e1-49fb-87f6-bff6d5d3bae3
# ╠═97b02a67-4eb2-4374-8939-f947eeddf637
# ╠═abccc425-4ac5-4285-80e6-ac7b2390993c
# ╠═cc7d6f03-045d-4a8a-a2e6-ec51bb0cc4a6
# ╠═a864be88-ba2a-40a6-bcc0-e5147a79b668
# ╟─c33b7c24-283f-4859-97c0-d225c7971633
# ╠═af2a3500-4120-476f-8b00-3ba3d0867977
# ╠═4b15287b-3df2-4603-b909-37b6051bf76f
# ╠═4c38e421-c314-4931-85de-2f6079955d7b
# ╟─c1483996-5ddc-400c-81ee-07249b5f4619
# ╠═06801aec-662e-450f-868b-12c7be4dd7b1
# ╠═9b565ae6-2af9-460f-b7d1-d0e999a37d0b
# ╟─fad73216-dbc6-4eb5-9d0b-1a37091af5fa
# ╟─ff80c05d-f286-4c4c-9228-453034e2e823
# ╠═6e4f8b25-a012-4138-a505-1698d15ed348
# ╠═b80ff287-29b6-45e0-8910-3f36dbc1e94e
# ╟─9038c6d8-fdf4-4228-8689-8000a0948b49
# ╠═a8fc1b81-648a-45c3-8207-9aa6452f34bd
# ╠═ea699478-d4b9-4585-af2d-4d42ff59f63d
# ╟─4431618d-9519-4ad4-adde-9891b5bea9e0
# ╠═08bd7450-c786-48e7-8bd7-4358a0505b65
# ╟─6d2e85c1-14dc-4698-b1f0-8d705d699e44
# ╠═75dd21ea-d3ab-4e56-886f-581baad8ec88
# ╟─547bbbf7-3d64-4277-83fc-a127c7f03c1e
# ╠═2727be0b-5bee-4bd8-a744-07ab5901fe44
# ╠═8eccda41-939e-45e3-9762-6298704d15c0
# ╠═1e04b229-6be9-4708-aa8e-c46de78d1c9c
# ╟─427234d0-be43-47c0-93e1-4e3c7af09e58
# ╠═6192af5b-e405-4532-ae43-200dc199de9d
# ╟─2b65cc6a-5059-4cc8-ae91-867ab5ec7d2f
# ╠═62bdffa1-1fbd-46fb-9696-83ada184f77f
# ╠═5e70c188-04c7-4464-8ed2-f1c2c067abd0
# ╠═80eb9a45-ac2b-4243-8be6-d49bab32d079
# ╟─4d8d55bf-c9e8-433d-90bf-6ef5b51cce7d
# ╠═9dde4458-1044-4cdb-965d-32bbc39fdf8f
# ╠═99aa4c2b-862a-4ba3-b8ff-418dda74b4fb
# ╠═b1433e12-6a9c-4324-9811-a3bd3e71ca0f
# ╠═e24d5069-3053-4c33-8855-e218c9f0d85a
# ╟─d5ae4a57-cb05-417c-9a31-97e77d657fbe
# ╠═27fe7d76-9525-41b6-b2e6-6e991e3abdaa
# ╠═442723b0-36a3-4855-bf09-5799c78cf716
# ╠═fe4fa23e-945a-4264-bec9-bb09a450616d
# ╟─465625c7-d0b9-4f6a-91e8-4ec092d891de
# ╠═c67d5547-b8fd-46e3-8ca8-47e851da1428
# ╟─e11a55d2-f14a-464e-9c2e-67a2fef30e89
# ╠═b233ff17-855d-4ac9-b06d-8fe483bb4d63
# ╟─73483cf3-3e48-4ca9-9b37-e4433ef01799
# ╟─0a260c90-476d-4190-be50-7019a01c317a
# ╠═b7a92f45-195a-4301-9ea5-3bfbdfb351ac
# ╟─35cb458b-56cd-426b-acd4-4932c380498e
# ╟─8b0f8b43-7233-422b-a579-6a4e8df1c77a
# ╠═ea77cbb4-11fc-460f-9593-c12ecb06ba45
# ╠═5d395d27-221a-48a6-9567-70d7cf636705
# ╠═cf7caf2c-146d-424f-beb6-a2a8cef12321
# ╠═2ebcf881-35dc-4f03-98bf-39b014535805
# ╠═2659f1db-9a27-4bd0-a471-0c38824e659f
# ╠═e495d916-dcb2-471c-9c7c-dc61aecb580e
# ╠═aed984b7-dd7b-474b-bb78-ee0397e681b3
# ╟─9eab56e3-68e5-40df-b403-b66da5e71b17
# ╠═019912a7-fdcb-4776-bf0f-913dca46dafd
# ╠═b5eb8829-cb85-400f-ba7d-5b87521d4ba6
# ╟─ccb1dcdc-522e-4c0d-be2c-a3f1eb19348c
# ╠═97103d90-8327-452a-be66-3be646968689
# ╠═7dd8344d-4f7d-4a84-83f8-3a44a1cc88e7
# ╠═d0bac7ad-d90f-4001-917c-27605c365c03
# ╟─0882a6f7-f1d4-4f72-a1b2-a43524ab715c
# ╠═40cddda3-acbb-47e9-ab57-939d3c12a6c6
# ╠═07a07350-0156-415b-bfdc-ca55da2cd68b
# ╠═fa2bc10b-8d0b-4993-86cc-4e7cec9a44a6
# ╠═701235ab-0d24-4e1c-952d-bbb8f3f61be2
# ╟─1edb759a-48a1-4e98-bc38-563c8d1680a7
# ╠═cabd0588-e36f-4fe4-9db3-0403d25c656a
# ╟─84be58fe-41ab-4d1a-8242-03ad68e6a952
# ╠═8f65f71b-3fd1-45a1-92e1-93baae1e8e4d
# ╠═b8a550bf-8381-44ba-be92-5805d84b7f98
# ╟─513c10a1-31f0-4c7d-8102-1cbab98a84ea
# ╠═aae3ac16-1c4b-4a7b-8039-713d98fd9ea5
# ╠═a215bf4e-8d96-4cbb-8407-29b2907f3d5c
# ╟─85ca81bc-030c-4ce2-a486-e38923f85368
# ╠═ce3d30d6-d8b7-4ae4-86f0-3a2fce3a1bca
# ╠═33ca2228-7962-4818-908a-12a346cdbc86
# ╟─99b8bd9e-0045-485d-a53a-b021199536ea
# ╠═2a319b11-d7e5-4b50-a689-46122514bc81
# ╠═5ec7d923-b2cd-4690-bd14-448a1c73d12c
# ╟─23e10e87-b78c-422f-8c31-a5ab0647a13d
# ╠═4dca08e3-0cfc-4b2c-8b05-a4bf4ada3458
# ╠═2610b1f7-180a-4732-a0dd-597a0685ddda
# ╠═42b5c883-4849-45fc-8a81-fe48dcbaeffc
# ╠═2468dc1f-8721-403e-b195-cdd3d0da3558
# ╟─14a0eae8-be5d-4652-b187-6c3cfd629c62
# ╠═6d93ac7a-3634-4ab8-8aaa-311e5a2d657a
# ╟─1b59151d-6b90-4df4-bc96-3997272cd8d7
# ╠═0c4066c0-5ee7-4029-9e88-3b96bde0b19d
# ╠═8e110723-c9fe-4a65-8a9c-1ed7b2b1c5f6
# ╟─211d4a37-ba55-4514-8cf0-27e665050c71
# ╟─1dcec520-4e80-4403-ae2a-5e096c39d7f4
# ╠═e6c6c0c2-d4c6-43d9-a57b-99f0faa8dd0f
# ╠═da4585c8-254a-444e-83ea-d8a205840055
# ╠═4e46f10a-69f5-4ee3-a7f9-7fe1f13e7391
# ╠═b62b8ca5-e746-43b7-a1ae-fd5f28a83f28
# ╟─28d0687f-97ef-44df-beb4-aa54af665482
# ╠═e6ab08be-4977-409b-a33a-382e8fcdb244
# ╟─2f43036d-d3cd-467b-bb27-5616cad0ca52
# ╠═0592f7c7-4239-4bb3-9c0b-a5779746d7df
# ╠═78d56dba-2c4a-41a5-acc7-7a3f48c6b207
# ╠═70b093de-f991-49fe-a909-8082a6cf9bc0
# ╟─5c9830e2-b789-425f-8f1d-0952fab05c8a
# ╟─f2ba7898-d589-4112-87b0-7fc8817a6b12
# ╠═66860084-849d-48ce-8273-fc10d31612fb
# ╟─e2033d92-cf0c-4963-a44e-38a2ed8ddb10
# ╟─04ab9804-2437-4696-bb72-ecaee95213a9
# ╠═2480e4cd-393f-48d8-8ec7-22d8e381c560
# ╠═25b80a64-d9e1-43a6-916b-e43755c252f8
# ╠═3a387f24-7ee6-405c-8c06-d6efe2207a6f
# ╠═037baedd-0fae-4098-afaf-e516268d7b81
# ╟─fa1230b0-8919-4f15-86a9-8b19627f3809
# ╠═40865cf5-6d91-4d15-90a7-738f8d3ce18f
# ╠═3c1f1fd8-9db0-4fdc-b90b-37e81319e606
# ╟─3804859c-ab83-4929-9b8b-b34961838d1f
# ╠═03a4a6bc-f37c-4a2b-ba78-5661ce6519b5
# ╠═9bee7432-3ad9-4d01-9874-ef545cd5351d
# ╠═bc9f297d-3035-4b7c-90d7-37773103f61f
# ╟─a5d138b8-ec35-4734-89de-5743be03cc4c
# ╠═79921d4e-af5e-448f-a42e-f63a4a65dd83
# ╠═e4f5223a-655b-4c44-91ed-955cfd43a2d8
# ╠═f3ed4faa-aa3b-41fa-b755-b14c745c2692
# ╟─99e69935-02dc-4daf-9576-ffc73dad4a2a
# ╠═8c5a123f-dc5a-40e1-aa91-c8bb77e61527
# ╟─588c50a2-2a1b-4362-804a-9773e880f537
# ╠═8d73e153-34fd-439c-84fe-5438ef589790
# ╟─136e37ca-52b9-4158-bf67-ea577b463754
# ╟─d3e5df42-49d7-4c3b-b00f-ae296a104258
# ╠═6b59d6b1-fdbe-496f-9083-55da8d62c393
# ╟─4321c5ae-23bb-47d5-b83d-1dbaced64edc
# ╟─9dd546c7-0e84-4319-86cd-a4cb0e6291be
# ╠═66ffb748-e9b6-4833-a684-c7ccce08b622
# ╠═442278ec-e29f-4c83-bb8a-ac5d27b28e95
# ╠═3ad081a4-a133-47df-8b0a-20173ff33d84
# ╠═0a3b57b9-98fc-41d1-8238-f8079df849c6
# ╠═e7b131a1-948e-4b7b-853b-bbcbb53f2ec9
# ╠═bd271f8b-0b47-4010-9606-d1a37318a7b5
# ╟─0d9ce8a8-fa5b-43f5-8a32-eac088b2b428
# ╠═fb324892-8995-4c51-9d6c-0f4eea2ed1e9
# ╠═b881ed38-7f06-4561-86fd-1f6d1d9362cf
# ╠═b83c2287-14ec-4892-bf0f-c5898f571268
# ╠═fbfb3fd8-8b45-44a7-a2bd-aedc8b823331
# ╟─bc90ea38-0853-40b5-bae8-31186da91dce
# ╠═189ed23d-5903-4461-aea2-17183939b1f9
# ╟─fbdeb6fb-4e1c-482c-871e-f2fccdf96687
# ╠═94f7527d-0546-4713-aeb7-71dc66f1c17f
# ╠═732f6bd9-da0f-414a-aec1-2bb225c302b9
# ╠═dc67c9f3-5553-4cf6-8233-070bd04d3c10
# ╠═323f1256-bcde-4b07-a374-04e075980dae
# ╠═19e533c9-7bf7-4710-ae27-ba419c65f368
# ╟─f2a668a5-5f1e-41eb-9cf0-212960c3da49
# ╠═d5c95557-9efa-43ef-a6c6-12092109cb68
# ╟─0d635435-7350-4afc-a061-7857d106c989
# ╠═dd3c6efd-9344-4a52-a681-59922f3bac6f
# ╟─7052fa4f-cb75-4962-ab98-4f1c025f2355
# ╟─78ef14fa-7a3f-49a1-b5c9-3dd8d7d30881
# ╟─8d69cfbb-94bc-4747-87f8-647c53f9ae6d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
