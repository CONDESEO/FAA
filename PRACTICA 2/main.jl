
import Pkg
Pkg.activate(".")


println("Iniciando ejecución completa del proyecto...")

# -------------------------
# 1. Preprocesamiento
# -------------------------
println("Cargando y preprocesando los datos...")
include("preprocesamiento_datos.jl")  # genera: inputs, targets, clases

# -------------------------
# 2. Generar índices de CV
# -------------------------
println("Generando índices de validación cruzada...")
include("modelo_utils.jl")
using Random, CSV, DataFrames
seed!(1234)
cv_idx = crossvalidation(targets, 5)
CSV.write("indices_cv5.csv", DataFrame(fold = cv_idx))
println("indices_cv5.csv generado correctamente.")

# -------------------------
# 3. Entrenamiento de modelos
# -------------------------
include("entrenar_modelos.jl")

# -------------------------
# 4. Visualización de resultados
# -------------------------
include("visualizar_resultados.jl")  # este debería usar `resultados` y `clases`
