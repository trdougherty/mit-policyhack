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
end

# ╔═╡ 687255fd-80db-4e71-99f9-bde74265fa97
census_data = CSV.read("R13210086_SL140.csv", DataFrame, skipto=3, types=Dict(:FIPS => String))

# ╔═╡ c4df97ac-079c-443f-8f93-ce91f24ee68c
filings_information = leftjoin(evictionsᵧ, census_data, on=:FIPS)

# ╔═╡ 97dd2be3-8511-4d24-ad8a-c7109158ec39
tree_class = @load RandomForestRegressor pkg=DecisionTree verbosity=0

# ╔═╡ d461b21c-e6a6-450a-a10a-d7e22c3e5fea
tree_model = tree_class()

# ╔═╡ 4718e5a5-bdec-41b3-a2e5-a29dbd481c33
keeping_cols = names(
	select(
		filings_information, 
		Not(
			[
				"filings_avg",
				"Nation",
				"State",
				"County",
				"Census Tract", 
				"Total Population_1",
				"Total Population:",
				"Total Population:_1"
			]
		)
	), Union{Real, Missing}
);

# ╔═╡ 31aa281e-3650-4bd9-8a55-42464998d40a
keeping_colsᵦ = filter( x -> 
	~occursin("Renter-Occupied Housing", x) &&
	x != "Median Gross Rent" &&
	x != "Median Value",
	keeping_cols
);

# ╔═╡ 40180fb4-ade0-44a9-81d8-7e20646ccb04
keeping_cols

# ╔═╡ a978b7a8-92e6-4976-b1b5-362034026506
X = select(filings_information, keeping_colsᵦ);

# ╔═╡ 8a41663e-74d3-4d6b-8cd8-bcb01f3ac170
missing_idx₀ = completecases(X)

# ╔═╡ d0f2e7b2-31cc-412e-86c8-3287a810ef26
begin
X′ = X[missing_idx₀, :];
Y′ = filings_information[missing_idx₀,:].filings_avg;
end;

# ╔═╡ e5157a23-b98f-4d1a-a339-1c5fb58542df
tree = machine(tree_model, X′, Y′)

# ╔═╡ 3026c5b7-91b7-43c7-89a2-398abe1842e5
train, test = partition(eachindex(Y′), 0.8, shuffle=true); # 70:30 split

# ╔═╡ 11cc822a-8873-4586-8bea-e277c0528a1d
fit!(tree, rows=train)

# ╔═╡ 95d9155f-62ca-463b-a5fd-a36294e8d058
Ŷ = predict(tree, X′[test,:]);

# ╔═╡ b4a55958-fd6f-4db5-aea1-0df005456f44
# first what would a random guess look like
null = repeat([mean(Y′[train])], length(test));

# ╔═╡ 6e86b1cc-31a6-4097-8843-f2a8cd8bdde3
rmse(Y′[test], null)

# ╔═╡ d9bbce21-8014-4f27-abc9-a2009dbd5d0b
rmse(Y′[test], Ŷ)

# ╔═╡ 29cbbaf6-ec8c-4a4b-8cc3-9d5967fa2443


# ╔═╡ beb6bdd4-010f-4c19-9b59-577ce69259bc
# now to make a prediction for all of the census data
Xα = select(census_data, keeping_colsᵦ)

# ╔═╡ 05f9779d-2d96-4479-92fc-b52a8817d58e
missing_idx = completecases(Xα)

# ╔═╡ 04fbedfe-c4f9-4897-bf52-12f9ba8f9bad
Xα′ = Xα[missing_idx, :];

# ╔═╡ 4bded0c7-d9ec-47c8-b272-26bf6c290130
nrow(Xα)

# ╔═╡ f21916c4-80a5-47e1-8562-04cbd5665dbb
nrow(Xα′)

# ╔═╡ 3c182763-1e77-4744-89d0-4e74af18921c


# ╔═╡ e05b40dd-e363-473e-8e08-410be3aa79fe
Yα = predict(tree, Xα′)

# ╔═╡ d9a1a60a-afbc-4676-8e7e-b099f20a9f6d
census_predictiondata = census_data[missing_idx, :];

# ╔═╡ 3ef9394e-ee6c-4ad2-9af0-8e327827cb42
census_predictiondata[:, "predicted_fillings"] = Yα;

# ╔═╡ 3f4a45ef-6bc2-4a0d-a5ca-af331763ca44
censusⱼ = rename(select(census_predictiondata, ["FIPS","predicted_fillings"]), "FIPS" => "GEOID")

# ╔═╡ 55ced503-d652-4289-8910-691371bd85d9
CSV.write("census_eviction_track_predictions.csv", censusⱼ)

# ╔═╡ Cell order:
# ╠═d50eccaf-16c9-422f-a355-7321495f1b4b
# ╠═90d129f7-e32b-4e68-b922-e3312478d315
# ╠═e04f8a3a-4c2f-11ed-0488-97447fd26510
# ╠═8d2e80a8-61f3-47d9-ac35-bcd13b39d142
# ╠═687255fd-80db-4e71-99f9-bde74265fa97
# ╠═c4df97ac-079c-443f-8f93-ce91f24ee68c
# ╠═97dd2be3-8511-4d24-ad8a-c7109158ec39
# ╠═d461b21c-e6a6-450a-a10a-d7e22c3e5fea
# ╠═4718e5a5-bdec-41b3-a2e5-a29dbd481c33
# ╠═31aa281e-3650-4bd9-8a55-42464998d40a
# ╠═40180fb4-ade0-44a9-81d8-7e20646ccb04
# ╠═a978b7a8-92e6-4976-b1b5-362034026506
# ╠═8a41663e-74d3-4d6b-8cd8-bcb01f3ac170
# ╠═d0f2e7b2-31cc-412e-86c8-3287a810ef26
# ╠═e5157a23-b98f-4d1a-a339-1c5fb58542df
# ╠═3026c5b7-91b7-43c7-89a2-398abe1842e5
# ╠═11cc822a-8873-4586-8bea-e277c0528a1d
# ╠═95d9155f-62ca-463b-a5fd-a36294e8d058
# ╠═b4a55958-fd6f-4db5-aea1-0df005456f44
# ╠═6e86b1cc-31a6-4097-8843-f2a8cd8bdde3
# ╠═d9bbce21-8014-4f27-abc9-a2009dbd5d0b
# ╠═29cbbaf6-ec8c-4a4b-8cc3-9d5967fa2443
# ╠═beb6bdd4-010f-4c19-9b59-577ce69259bc
# ╠═05f9779d-2d96-4479-92fc-b52a8817d58e
# ╠═04fbedfe-c4f9-4897-bf52-12f9ba8f9bad
# ╠═4bded0c7-d9ec-47c8-b272-26bf6c290130
# ╠═f21916c4-80a5-47e1-8562-04cbd5665dbb
# ╠═3c182763-1e77-4744-89d0-4e74af18921c
# ╠═e05b40dd-e363-473e-8e08-410be3aa79fe
# ╠═d9a1a60a-afbc-4676-8e7e-b099f20a9f6d
# ╠═3ef9394e-ee6c-4ad2-9af0-8e327827cb42
# ╠═3f4a45ef-6bc2-4a0d-a5ca-af331763ca44
# ╠═55ced503-d652-4289-8910-691371bd85d9
