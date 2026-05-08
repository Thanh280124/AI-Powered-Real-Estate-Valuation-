import pandas as pd

df = pd.read_csv("data/ames_housing.csv")

filtered = df[
    (df["Neighborhood"] == "NridgHt") &
    (df["OverallQual"] == 9) &
    (df["GrLivArea"].between(2450 * 0.85, 2450 * 1.15))
]

print(filtered[["Neighborhood", "GrLivArea", "OverallQual", "YearBuilt", "SalePrice"]])
