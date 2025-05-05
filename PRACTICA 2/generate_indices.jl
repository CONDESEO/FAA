# generate_indices.jl

using CSV, DataFrames, Random

# 1) Incluye tu preprocesado para tener `inputs` y `targets` en memoria
include("preprocessing.jl")

# 2) Fija semilla para reproducibilidad
seed!(1234)

# 3) Genera el vector de folds
cv_idx = crossvalidation(targets, 5)

# 4) Guarda en CSV
CSV.write("indices_cv5.csv", DataFrame(fold=cv_idx))

println("✅ indices_cv5.csv generado correctamente.")
