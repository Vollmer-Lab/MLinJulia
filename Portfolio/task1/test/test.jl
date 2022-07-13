using Test

include("../src/sample.jl")

using Main.BradleyTerry

function Base.isapprox(l::Dict{Any,Float64}, r::Dict{Any,Float64})
    if length(l) != length(r)
        return false
    end
    for (key, value) in l
        if !haskey(r, key)
            return false
        elseif !Base.isapprox(value, r[key], atol=0.0001)
            return false
        end
    end
    true
end

@testset "Simple example dataset: beverage preferences" begin
    example = [("tea", "coffee"), ("tea", "coffee"),
        ("tea", "coffee"), ("tea", "coffee"), ("tea", "coffee"), 
        ("coffee", "tea"), ("chocolate", "coffee"), ("tea", "chocolate"),
        ("chocolate", "coffee")]
    scores = BradleyTerryModelWrapper(example)
    # The following 'magic numbers' were calculated using BradleyTerry2 in R.
    # Careful of the ordering of the output scores.
    expected = Dict("chocolate" => 0.312544, "coffee" => -1.21582, "tea" => 0.903272)
    @test isapprox(scores, expected)
end

using CSV, DataFrames

@testset "Reproduce journals ranking from Varin et al (2016) citation data" begin
    citations = CSV.read("stats_journals.csv", DataFrame) # format: winner, loser, n
    expected = CSV.read("stats_journal_scores.csv", DataFrame, types = [String, Float64])
    expected = Dict(zip(expected.journal, expected.mu))
    scores = BradleyTerryModelWrapper(citations)
    @test isapprox(scores, expected)
end