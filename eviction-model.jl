### A Pluto.jl notebook ###
# v0.19.13

using Markdown
using InteractiveUtils

# ╔═╡ d50eccaf-16c9-422f-a355-7321495f1b4b
using Pkg

# ╔═╡ 90d129f7-e32b-4e68-b922-e3312478d315
Pkg.activate(Base.current_project())

# ╔═╡ e04f8a3a-4c2f-11ed-0488-97447fd26510
begin
	using CSV
	using DataFrames
	using MLJ
	using Dates
	using Gadfly
end

# ╔═╡ 8d2e80a8-61f3-47d9-ac35-bcd13b39d142
begin
	evictions = CSV.read("evictions.csv", DataFrame)
	evictions[!,"month"] = Date.(evictions.month, dateformat"m/y")
	evictions[!, "year"] = Dates.year.(evictions.month)
	evictionsᵧ = combine(groupby(evictions, [:GEOID]), :filings_avg => first, renamecols=false)
	rename!(evictionsᵧ, :GEOID => :FIPS)
	evictionsᵧ = evictionsᵧ[evictionsᵧ.FIPS .!= "sealed", :]

	evictionsₑ = combine(groupby(evictions, :GEOID), :filings_2020 => mean, renamecols=false)
	rename!(evictionsₑ, :GEOID => :FIPS)
	evictionsₑ = evictionsₑ[evictionsₑ.FIPS .!= "sealed", :]
end

# ╔═╡ ac946678-5eb8-4b50-a48d-21a2f0ebf2ba
maximum(evictions.filings_avg)

# ╔═╡ 72e6f190-7684-44b9-9a21-cf73114af150
# evictionsᵧ[!,"Census Tract"] = map( x -> x[1:5], evictionsᵧ.FIPS)

# ╔═╡ 687255fd-80db-4e71-99f9-bde74265fa97
census_data = CSV.read("R13210361_SL140.csv", DataFrame, skipto=3, types=Dict(:FIPS => String))

# ╔═╡ bd0e2820-ec39-4fef-8839-02067b67e36a
census_description = describe(census_data, :nmissing, cols=names(census_data, Union{Float64, Int, Missing}))

# ╔═╡ 08b27fa3-6452-47b8-97d2-cf90a6456fcd
nrow(census_data)

# ╔═╡ b72c0971-1fa1-4134-94fd-12c18aa4ded1
describe(dropmissing(census_data, 
	["Median Gross Rent as a Percentage of Household  Income in the Past 12 Months (Dollars)"],
))

# ╔═╡ 7dab5614-7198-487c-b86e-2ea549f978f6
keep_names = names(census_data, Union{Real, Missing})[census_description.nmissing .< 4000]

# ╔═╡ c4df97ac-079c-443f-8f93-ce91f24ee68c
filings_information = leftjoin(evictionsᵧ, census_data, on=:FIPS)

# ╔═╡ a9babe29-aa07-4a3e-bce0-6841e0a58c8b
filings_informationₑ = leftjoin(evictionsₑ, census_data, on=:FIPS)

# ╔═╡ 97dd2be3-8511-4d24-ad8a-c7109158ec39
tree_class = @load RandomForestRegressor pkg=DecisionTree verbosity=0

# ╔═╡ d461b21c-e6a6-450a-a10a-d7e22c3e5fea
tree_model = tree_class()

# ╔═╡ b2305a4d-69c0-4c04-a8f2-6a7fb9e782dd
describe(census_data[:,keep_names], :nmissing)

# ╔═╡ 31aa281e-3650-4bd9-8a55-42464998d40a
begin
	keeping_colsᵦ = filter( x -> 
		~in(x, [
			"State (FIPS Code)",
			"County of current residence",
			"Logical Record Number",
			"Summary Level",
			"Geographic Component",
			"Census Tract",
		]) && ~occursin(r"_[0-9]", x),
		keep_names
	);
end

# ╔═╡ 5505412a-8920-4208-8839-b88d1bc98a6a
keeping_colsᵦ

# ╔═╡ 5c4dd5fc-5648-4bd6-9e06-f8f6047aa770


# ╔═╡ a978b7a8-92e6-4976-b1b5-362034026506
X = select(filings_information, keeping_colsᵦ);

# ╔═╡ de64eb52-881f-431c-be13-daac242b1c4f
Xₑ = select(filings_informationₑ, keeping_colsᵦ);

# ╔═╡ 8a41663e-74d3-4d6b-8cd8-bcb01f3ac170
missing_idx₀ = completecases(X);

# ╔═╡ 087f95e2-6f44-4006-8ae1-9ae540492a51
missing_idx₀ₑ = completecases(Xₑ);

# ╔═╡ d0f2e7b2-31cc-412e-86c8-3287a810ef26
begin
X′ = dropmissing(X[missing_idx₀, :]);
Y′ = filings_information[missing_idx₀,:].filings_avg;
end;

# ╔═╡ c60610aa-858d-4c91-a454-e11e3126ecbf
begin
X′ₑ = dropmissing(X[missing_idx₀ₑ, :]);
Y′ₑ = filings_informationₑ[missing_idx₀ₑ,:].filings_2020;
end;

# ╔═╡ 2fc4ee8d-5ad5-4ea8-a9e3-f6309803a4ef
filter( x -> occursin(r"rent"i, x), names(census_data))

# ╔═╡ 1edb7182-bb71-49ea-b885-9d6cee83c8a8
filter( x -> occursin("rent", x), names(filings_informationₑ, Union{Real, Missing}))

# ╔═╡ 345f2c66-0050-49a3-b4c2-5bd42a44bb04
Gadfly.plot(
	Gadfly.layer(
		x=Y′ₑ,
		Geom.histogram(bincount=100, density=true),
		Theme(default_color="lightblue", alphas=[0.6])
	),
	Gadfly.layer(
		x=Y′,
		Geom.histogram(bincount=100, density=true),
		Theme(default_color="indianred", alphas=[0.6])
	),
	Coord.Cartesian(xmin=0,xmax=20),
	Guide.title("Histogram of Eviction Rates"),
	Guide.manual_color_key("Legend", ["COVID", "Typical"], ["lightblue", "indianred"])
)

# ╔═╡ e5157a23-b98f-4d1a-a339-1c5fb58542df
tree = machine(tree_model, X′, Y′)

# ╔═╡ 3026c5b7-91b7-43c7-89a2-398abe1842e5
train, test = partition(eachindex(Y′), 0.8, shuffle=true); # 70:30 split

# ╔═╡ 11cc822a-8873-4586-8bea-e277c0528a1d
fit!(tree, rows=train)

# ╔═╡ 55894d93-6440-4de0-95b3-0a717b7370eb
feature_importances(tree)

# ╔═╡ 95d9155f-62ca-463b-a5fd-a36294e8d058
Ŷ = predict(tree, X′[test,:]);

# ╔═╡ 7ca900cf-8b43-4fbc-9af6-17a3672389ff


# ╔═╡ 76e6ca8f-11bc-470d-bf70-9c11f406aaa5
treeₑ = machine(tree_model, X′ₑ, Y′ₑ);

# ╔═╡ 0be7ea94-c2fc-4d3a-968b-09c812ce2f2d
feature_importances(treeₑ)[1:10]

# ╔═╡ 43cc7ce1-be10-477b-9f8f-d4c041c947de
trainₑ, testₑ = partition(eachindex(Y′ₑ), 0.8, shuffle=true); # 70:30 split

# ╔═╡ 723e6f50-a499-4458-b961-42bc1961805d
fit!(treeₑ, rows=trainₑ)

# ╔═╡ cd86336c-4d3a-4753-abdb-cbd3c6ca7f25
Ŷₑ = predict(treeₑ, X′ₑ[testₑ,:]);

# ╔═╡ d4ebe7fa-a5d5-4074-b0d9-d96f3e6db298


# ╔═╡ b4a55958-fd6f-4db5-aea1-0df005456f44
# first what would a random guess look like
null = repeat([mean(Y′[train])], length(test));

# ╔═╡ 6e86b1cc-31a6-4097-8843-f2a8cd8bdde3
rmse(Y′[test], null)

# ╔═╡ d9bbce21-8014-4f27-abc9-a2009dbd5d0b
rmse(Y′[test], Ŷ)

# ╔═╡ beb6bdd4-010f-4c19-9b59-577ce69259bc
# now to make a prediction for all of the census data
Xα = select(census_data, keeping_colsᵦ)

# ╔═╡ 05f9779d-2d96-4479-92fc-b52a8817d58e
missing_idx = completecases(Xα)

# ╔═╡ 04fbedfe-c4f9-4897-bf52-12f9ba8f9bad
Xα′ = Xα[missing_idx, :];

# ╔═╡ 6eb999d0-3eb8-45db-9b56-eff3c91615c3
describe(Xα′)

# ╔═╡ 4bded0c7-d9ec-47c8-b272-26bf6c290130
nrow(Xα)

# ╔═╡ f21916c4-80a5-47e1-8562-04cbd5665dbb
nrow(Xα′)

# ╔═╡ 56a97954-f455-4bcb-9cad-d837a1961d63
length(Ŷₑ)

# ╔═╡ e3ee2d58-9ab1-4955-93cf-aee4e86822d3
names(X′)

# ╔═╡ 3c182763-1e77-4744-89d0-4e74af18921c
names(Xα′)

# ╔═╡ 82f2f0ac-4d05-4417-b341-64bab93eb2d3
Xαₙ′ = dropmissing(Xα′);

# ╔═╡ e05b40dd-e363-473e-8e08-410be3aa79fe
Yα = predict(tree, Xαₙ′)

# ╔═╡ 85bdc227-2785-47dd-8820-d398fe2d0f1e
Yαₑ = predict(treeₑ, Xαₙ′)

# ╔═╡ d9a1a60a-afbc-4676-8e7e-b099f20a9f6d
census_predictiondata = census_data[missing_idx, :];

# ╔═╡ 3ef9394e-ee6c-4ad2-9af0-8e327827cb42
census_predictiondata[:, "predicted_filings"] = Yα;

# ╔═╡ 9cdfa278-ba57-4436-b8bd-3d5249c3b128
census_predictiondata[:, "predicted_extreme_filings"] = Yαₑ;

# ╔═╡ cd474028-fbc2-411b-b787-dcceab96724e
census_predictiondata.extreme_deviations = census_predictiondata.predicted_extreme_filings .- census_predictiondata.predicted_filings

# ╔═╡ a4869ba8-99ea-4fe4-b492-90a15ec4cc28
census_predictiondata.eviction_change = (census_predictiondata.extreme_deviations .* census_predictiondata[:,"Occupied Housing Units: Renter Occupied"]) ./ census_predictiondata[:,"Total Population"]

# ╔═╡ 7f7efa5a-11db-4178-bd3d-287c7a07ce1d
names(census_predictiondata)

# ╔═╡ 3f4a45ef-6bc2-4a0d-a5ca-af331763ca44
censusⱼ = rename(select(census_predictiondata, ["FIPS","eviction_change", "predicted_filings", "Total Population"]), "FIPS" => "GEOID")

# ╔═╡ aaa39303-1a13-4cf6-bfd1-3a0885355aaa


# ╔═╡ 55ced503-d652-4289-8910-691371bd85d9
CSV.write("census_eviction_track_predictions.csv", censusⱼ)

# ╔═╡ Cell order:
# ╠═d50eccaf-16c9-422f-a355-7321495f1b4b
# ╠═90d129f7-e32b-4e68-b922-e3312478d315
# ╠═e04f8a3a-4c2f-11ed-0488-97447fd26510
# ╠═8d2e80a8-61f3-47d9-ac35-bcd13b39d142
# ╠═ac946678-5eb8-4b50-a48d-21a2f0ebf2ba
# ╠═72e6f190-7684-44b9-9a21-cf73114af150
# ╠═687255fd-80db-4e71-99f9-bde74265fa97
# ╠═bd0e2820-ec39-4fef-8839-02067b67e36a
# ╠═08b27fa3-6452-47b8-97d2-cf90a6456fcd
# ╠═b72c0971-1fa1-4134-94fd-12c18aa4ded1
# ╠═7dab5614-7198-487c-b86e-2ea549f978f6
# ╠═c4df97ac-079c-443f-8f93-ce91f24ee68c
# ╠═a9babe29-aa07-4a3e-bce0-6841e0a58c8b
# ╠═97dd2be3-8511-4d24-ad8a-c7109158ec39
# ╠═d461b21c-e6a6-450a-a10a-d7e22c3e5fea
# ╠═b2305a4d-69c0-4c04-a8f2-6a7fb9e782dd
# ╠═31aa281e-3650-4bd9-8a55-42464998d40a
# ╠═5505412a-8920-4208-8839-b88d1bc98a6a
# ╠═5c4dd5fc-5648-4bd6-9e06-f8f6047aa770
# ╠═a978b7a8-92e6-4976-b1b5-362034026506
# ╠═de64eb52-881f-431c-be13-daac242b1c4f
# ╠═8a41663e-74d3-4d6b-8cd8-bcb01f3ac170
# ╠═087f95e2-6f44-4006-8ae1-9ae540492a51
# ╠═d0f2e7b2-31cc-412e-86c8-3287a810ef26
# ╠═c60610aa-858d-4c91-a454-e11e3126ecbf
# ╠═2fc4ee8d-5ad5-4ea8-a9e3-f6309803a4ef
# ╠═1edb7182-bb71-49ea-b885-9d6cee83c8a8
# ╠═345f2c66-0050-49a3-b4c2-5bd42a44bb04
# ╠═e5157a23-b98f-4d1a-a339-1c5fb58542df
# ╠═3026c5b7-91b7-43c7-89a2-398abe1842e5
# ╠═11cc822a-8873-4586-8bea-e277c0528a1d
# ╠═55894d93-6440-4de0-95b3-0a717b7370eb
# ╠═95d9155f-62ca-463b-a5fd-a36294e8d058
# ╠═7ca900cf-8b43-4fbc-9af6-17a3672389ff
# ╠═76e6ca8f-11bc-470d-bf70-9c11f406aaa5
# ╠═0be7ea94-c2fc-4d3a-968b-09c812ce2f2d
# ╠═43cc7ce1-be10-477b-9f8f-d4c041c947de
# ╠═723e6f50-a499-4458-b961-42bc1961805d
# ╠═cd86336c-4d3a-4753-abdb-cbd3c6ca7f25
# ╠═d4ebe7fa-a5d5-4074-b0d9-d96f3e6db298
# ╠═b4a55958-fd6f-4db5-aea1-0df005456f44
# ╠═6e86b1cc-31a6-4097-8843-f2a8cd8bdde3
# ╠═d9bbce21-8014-4f27-abc9-a2009dbd5d0b
# ╠═beb6bdd4-010f-4c19-9b59-577ce69259bc
# ╠═05f9779d-2d96-4479-92fc-b52a8817d58e
# ╠═04fbedfe-c4f9-4897-bf52-12f9ba8f9bad
# ╠═6eb999d0-3eb8-45db-9b56-eff3c91615c3
# ╠═4bded0c7-d9ec-47c8-b272-26bf6c290130
# ╠═f21916c4-80a5-47e1-8562-04cbd5665dbb
# ╠═56a97954-f455-4bcb-9cad-d837a1961d63
# ╠═e3ee2d58-9ab1-4955-93cf-aee4e86822d3
# ╠═3c182763-1e77-4744-89d0-4e74af18921c
# ╠═82f2f0ac-4d05-4417-b341-64bab93eb2d3
# ╠═e05b40dd-e363-473e-8e08-410be3aa79fe
# ╠═85bdc227-2785-47dd-8820-d398fe2d0f1e
# ╠═d9a1a60a-afbc-4676-8e7e-b099f20a9f6d
# ╠═3ef9394e-ee6c-4ad2-9af0-8e327827cb42
# ╠═9cdfa278-ba57-4436-b8bd-3d5249c3b128
# ╠═cd474028-fbc2-411b-b787-dcceab96724e
# ╠═a4869ba8-99ea-4fe4-b492-90a15ec4cc28
# ╠═7f7efa5a-11db-4178-bd3d-287c7a07ce1d
# ╠═3f4a45ef-6bc2-4a0d-a5ca-af331763ca44
# ╠═aaa39303-1a13-4cf6-bfd1-3a0885355aaa
# ╠═55ced503-d652-4289-8910-691371bd85d9
