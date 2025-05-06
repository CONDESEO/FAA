

# Fija semilla para reproducibilidad
seed!(1234)

# Genera el vector de folds
cv_idx = crossvalidation(targets, 5)

# Guarda en CSV
CSV.write("indices_cv5.csv", DataFrame(fold=cv_idx))

println("✅ indices_cv5.csv generado correctamente.")
