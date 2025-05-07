using CSV, DataFrames
using Statistics
using Random
using MLJ
using LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface
using MLJLIBSVMInterface: SVC

include("modelo_utils.jl")  # Solo este, el preprocesamiento ya se hizo en main.jl

# df_filtrado, player_dict, clases y folds ya deben estar definidos desde main.jl

# ----------------------------------
# Configuraciones para cada modelo
# ----------------------------------

#=
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
=#

model_configs = [
    (:ANN, Dict("topology" => [16], "learningRate" => 0.01, "minLoss" => 0.005, "numExecutions" => 1)),
    (:SVC, Dict("kernel" => "linear", "C" => 1.0)),
    (:DecisionTreeClassifier, Dict("max_depth" => 2)),
    (:KNeighborsClassifier, Dict("n_neighbors" => 3)),
    (:DoME, Dict("maximumNodes" => 2))
]


# ----------------------------------
# Ejecutar los experimentos
# ----------------------------------

resultados = DataFrame(
    Modelo = String[],
    Parametros = String[],
    AccuracyMedia = Float64[],
    AccuracyStd = Float64[]
)

println("Entrenando modelos y evaluando configuraciones...")

clases = String.(unique(df_filtrado.Winner))  # aseguras que sean Vector{String}
folds = Int64.(folds)

for (modelo, config) in model_configs
    println("\n Evaluando $modelo con parámetros: $(config)")
    
    if modelo == :ANN
        # Preprocesamiento global solo una vez
        inputs, encoded_targets = procesar_fold(df_filtrado, player_dict, clases; one_hot=true)
        
        acc_stats, _, _, _, _, _, _, _ = ANNCrossValidation(
            config["topology"],                      
            (inputs, encoded_targets),               
            folds;                                   
            numExecutions     = get(config, "numExecutions", 50),
            transferFunctions = get(config, "transferFunctions", fill(σ, length(config["topology"]))),
            maxEpochs         = get(config, "maxEpochs", 1000),
            minLoss           = get(config, "minLoss", 0.0),
            learningRate      = get(config, "learningRate", 0.01),
            validationRatio   = get(config, "validationRatio", 0),
            maxEpochsVal      = get(config, "maxEpochsVal", 20)
        )
    else
        
        acc_stats, _, _, _, _, _, _, _ = modelCrossValidation(
            modelo, config, df_filtrado, folds, player_dict, clases
        )
    end

    push!(resultados, (
        string(modelo),
        repr(config),
        round(acc_stats[1], digits=4),
        round(acc_stats[2], digits=4)
    ))
end

CSV.write("resultados.csv", resultados)
println("Resultados guardados en resultados.csv")
