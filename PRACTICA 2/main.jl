<<<<<<< Updated upstream
import Pkg
Pkg.activate(".")

println("Iniciando ejecución completa del proyecto...")

# -------------------------
# 1. Preprocesamiento inicial
# -------------------------
println("Cargando y preprocesando los datos...")
include("preprocesamiento_datos.jl")  # Genera: df_filtrado, clases, player_dict

# -------------------------
# 2. Generar índices de validación cruzada
# -------------------------
println("Generando índices de validación cruzada...")
include("modelo_utils.jl")
using Random, CSV, DataFrames
seed!(1234)
raw_targets = df_filtrado.Winner
folds = crossvalidation(raw_targets, 5)
CSV.write("indices_cv5.csv", DataFrame(fold = folds))
println("✔️ indices_cv5.csv generado correctamente.")

# -------------------------
# 3. Entrenamiento de modelos
# -------------------------
include("entrenar_modelos.jl")  # Usa los datos y folds ya generados

# -------------------------
# 4. Visualización de resultados
# -------------------------
include("visualizar_resultados.jl")  # Lee resultados.csv y genera gráficos
=======
   
>>>>>>> Stashed changes
