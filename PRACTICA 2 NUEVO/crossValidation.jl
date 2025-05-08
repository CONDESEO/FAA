using CSV, DataFrames, Random, MLJ
using LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface
using MLJLIBSVMInterface: SVC

include("modelUtils.jl")      # contiene funciones del profesor
include("preprocesado.jl")    # contiene la función procesar_fold()

# --------------------------
# Configuraciones de modelos
# --------------------------
model_configs = [
    (:ANN, Dict("topology" => [16], "learningRate" => 0.01, "minLoss" => 0.005, "numExecutions" => 1)),
    (:SVC, Dict("kernel" => "linear", "C" => 1.0)),
    (:DecisionTreeClassifier, Dict("max_depth" => 3)),
    (:KNeighborsClassifier, Dict("n_neighbors" => 5)),
    (:DoME, Dict("maximumNodes" => 4))
]

# --------------------------
# Definiciones del dataset
# --------------------------
df_filtrado = df                    # asumimos que ya está preprocesado
clases = [0, 1]                     # clasificación binaria 0 (ganó Player_2) o 1 (ganó Player_1)
player_dict = jugador_dict         # creado en main
folds = crossvalidation(df_filtrado.Winner, 5)

# --------------------------
# Validación cruzada
# --------------------------
resultados = DataFrame(Modelo = String[], Parametros = String[], AccuracyMedia = Float64[], AccuracyStd = Float64[])

for (modelo, config) in model_configs
    println("\nEvaluando $modelo con configuración: $(config)")

    # Vectores para métricas
    numFolds = maximum(folds)
    accuracies = Float64[]

    for fold_num in 1:numFolds
        df_train = df_filtrado[folds .!= fold_num, :]
        df_test  = df_filtrado[folds .== fold_num, :]

        # Preprocesado por fold
        X_train, y_train = procesar_fold(df_train, player_dict, clases; one_hot=(modelo == :ANN))
        X_test,  y_test  = procesar_fold(df_test, player_dict, clases; one_hot=(modelo == :ANN))

        if modelo == :ANN
            ann, _ = trainClassANN(config["topology"], (X_train, y_train);
                learningRate = get(config, "learningRate", 0.01),
                minLoss      = get(config, "minLoss", 0.0),
                maxEpochs    = get(config, "maxEpochs", 1000),
                transferFunctions = get(config, "transferFunctions", fill(σ, length(config["topology"])))
            )
            y_pred = classifyOutputs(ann(Float32.(X_test'))')
            y_true = y_test
        elseif modelo == :DoME
            y_pred = trainClassDoME((X_train, y_train), X_test, config["maximumNodes"])
            y_true = y_test
        else
            # MLJ (SVM, Árboles, kNN)
            if modelo == :SVC
                model = SVC(kernel = LIBSVM.Kernel.Linear, cost = config["C"])
            elseif modelo == :DecisionTreeClassifier
                DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0
                model = DecisionTreeClassifier(max_depth = config["max_depth"], rng = Random.MersenneTwister(1))
            elseif modelo == :KNeighborsClassifier
                KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels verbosity=0
                model = KNNClassifier(K = config["n_neighbors"])
            else
                error("Modelo no reconocido: $modelo")
            end

            mach = machine(model, MLJ.table(X_train), categorical(string.(y_train)))
            fit!(mach, verbosity=0)
            y_pred = mode.(MLJ.predict(mach, MLJ.table(X_test)))
            y_true = string.(y_test)
        end


        # Métrica
        acc = mean(y_pred .== y_true)
        push!(accuracies, acc)
    end

    push!(resultados, (
        string(modelo),
        repr(config),
        round(mean(accuracies), digits=4),
        round(std(accuracies), digits=4)
    ))
end

CSV.write("resultados.csv", resultados)
println("Resultados guardados en resultados.csv")
