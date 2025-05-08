using CSV, DataFrames
using Statistics
using Random
using MLJ
using LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface
using MLJLIBSVMInterface: SVC

include("modelo_utils.jl")

# Generar o cargar índices de validación cruzada
folds = crossvalidation(raw_targets, 5)
CSV.write("indices_cv5.csv", DataFrame(fold = folds))

# Definir configuraciones para cada modelo
model_configs = [
    # Redes neuronales
    (:ANN, Dict("topology" => [32], "learningRate" => 0.01, "minLoss" => 0.005)),
    (:ANN, Dict("topology" => [64], "learningRate" => 0.01, "minLoss" => 0.005)),
    (:ANN, Dict("topology" => [32, 16], "learningRate" => 0.005, "minLoss" => 0.005)),
    (:ANN, Dict("topology" => [64, 32], "learningRate" => 0.005, "minLoss" => 0.005)),
    (:ANN, Dict("topology" => [128, 64], "learningRate" => 0.001, "minLoss" => 0.005)),
    (:ANN, Dict("topology" => [32, 8], "learningRate" => 0.01, "minLoss" => 0.005)),
    (:ANN, Dict("topology" => [64, 16], "learningRate" => 0.005, "minLoss" => 0.005)),


    # SVM
    (:SVC, Dict("kernel" => "linear", "C" => 0.01)),
    (:SVC, Dict("kernel" => "linear", "C" => 0.1)),
    (:SVC, Dict("kernel" => "linear", "C" => 1.0)),
    (:SVC, Dict("kernel" => "rbf", "C" => 0.1)),
    (:SVC, Dict("kernel" => "rbf", "C" => 1.0)),
    (:SVC, Dict("kernel" => "poly", "C" => 1.0, "degree" => 2)),
    (:SVC, Dict("kernel" => "poly", "C" => 1.0, "degree" => 3)),
    (:SVC, Dict("kernel" => "sigmoid", "C" => 0.5)),

    # Árboles de decisión
    (:DecisionTreeClassifier, Dict("max_depth" => 2)),
    (:DecisionTreeClassifier, Dict("max_depth" => 4)),
    (:DecisionTreeClassifier, Dict("max_depth" => 6)),
    (:DecisionTreeClassifier, Dict("max_depth" => 8)),
    (:DecisionTreeClassifier, Dict("max_depth" => 10)),
    (:DecisionTreeClassifier, Dict("max_depth" => 12)),

    # kNN
    (:KNeighborsClassifier, Dict("n_neighbors" => 1)),
    (:KNeighborsClassifier, Dict("n_neighbors" => 3)),
    (:KNeighborsClassifier, Dict("n_neighbors" => 5)),
    (:KNeighborsClassifier, Dict("n_neighbors" => 7)),
    (:KNeighborsClassifier, Dict("n_neighbors" => 9)),
    (:KNeighborsClassifier, Dict("n_neighbors" => 11)),

    # DoME
    (:DoME, Dict("maximumNodes" => 2)),
    (:DoME, Dict("maximumNodes" => 3)),
    (:DoME, Dict("maximumNodes" => 4)),
    (:DoME, Dict("maximumNodes" => 5)),
    (:DoME, Dict("maximumNodes" => 6)),
    (:DoME, Dict("maximumNodes" => 8)),
    (:DoME, Dict("maximumNodes" => 10)),
    (:DoME, Dict("maximumNodes" => 12))
]

# Ejecutar experimentos y guardar resultados
resultados = DataFrame(
    Modelo = String[],
    Parametros = String[],
    AccuracyMedia = Float64[],
    AccuracyStd = Float64[]
)

println("Entrenando modelos y evaluando configuraciones...")

for (modelo, config) in model_configs
    println("\n Evaluando $modelo con parámetros: $(config)")
    acc_stats, _, _, _, _, _, _, _ = modelCrossValidation(modelo, config, (inputs, raw_targets), folds)
    push!(resultados, (
        string(modelo),
        repr(config),
        round(acc_stats[1], digits=4),
        round(acc_stats[2], digits=4)
    ))
end

CSV.write("resultados.csv", resultados)
println("Resultados guardados en resultados.csv")
