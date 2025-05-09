using CSV, DataFrames, Statistics, Random
import Statistics: std
using MLJ, LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface
using Flux
using Flux.Losses

Random.seed!(1234)

include("modelUtils.jl")

# Generamos los folds de validación cruzada
folds = crossvalidation(df.Winner, 5)
CSV.write("data/Archivo Indices.csv", DataFrame(Fold = folds))
k = 5


# Estructura para almacenar resultados por configuración
resultados_configuraciones = DataFrame(
    Modelo = String[],
    Parametros = String[],
    AccuracyMedia = Float64[],
    AccuracyStd = Float64[]
)

# Definimos las configuraciones para cada modelo
model_configs = [
    # ANN
    (:ANN, Dict("topology" => [5], "learningRate" => 0.01, "minLoss" => 0.01)),
    (:ANN, Dict("topology" => [10], "learningRate" => 0.01, "minLoss" => 0.01)),
    (:ANN, Dict("topology" => [20], "learningRate" => 0.01, "minLoss" => 0.01)),
    (:ANN, Dict("topology" => [50], "learningRate" => 0.005, "minLoss" => 0.01)),
    (:ANN, Dict("topology" => [10, 5], "learningRate" => 0.005, "minLoss" => 0.01)),
    (:ANN, Dict("topology" => [20, 10], "learningRate" => 0.005, "minLoss" => 0.01)),
    (:ANN, Dict("topology" => [10, 20], "learningRate" => 0.001, "minLoss" => 0.01)),
    (:ANN, Dict("topology" => [20, 5], "learningRate" => 0.01, "minLoss" => 0.01)),
    (:ANN, Dict("topology" => [5, 20], "learningRate" => 0.005, "minLoss" => 0.01)),
    (:ANN, Dict("topology" => [10, 10], "learningRate" => 0.01, "minLoss" => 0.01)),
    (:ANN, Dict("topology" => [30, 15], "learningRate" => 0.005, "minLoss" => 0.01)),
    (:ANN, Dict("topology" => [15, 30], "learningRate" => 0.001, "minLoss" => 0.01)),
    (:ANN, Dict("topology" => [20, 10, 5], "learningRate" => 0.005, "minLoss" => 0.01)),
    (:ANN, Dict("topology" => [10, 20, 10], "learningRate" => 0.001, "minLoss" => 0.01)),
    (:ANN, Dict("topology" => [5, 10, 5], "learningRate" => 0.01, "minLoss" => 0.01)),
    (:ANN, Dict("topology" => [30], "learningRate" => 0.01, "minLoss" => 0.005)),
    (:ANN, Dict("topology" => [40, 20], "learningRate" => 0.005, "minLoss" => 0.005)),
    (:ANN, Dict("topology" => [20, 40], "learningRate" => 0.001, "minLoss" => 0.005)),
    (:ANN, Dict("topology" => [15, 15], "learningRate" => 0.005, "minLoss" => 0.01)),
    (:ANN, Dict("topology" => [25, 10, 5], "learningRate" => 0.005, "minLoss" => 0.005)),

    # SVM
    (:SVC, Dict("kernel" => LIBSVM.Kernel.Linear, "C" => 0.01)),
    (:SVC, Dict("kernel" => LIBSVM.Kernel.Linear, "C" => 0.1)),
    (:SVC, Dict("kernel" => LIBSVM.Kernel.Linear, "C" => 1.0)),
    (:SVC, Dict("kernel" => LIBSVM.Kernel.RadialBasis, "C" => 0.1)),
    (:SVC, Dict("kernel" => LIBSVM.Kernel.RadialBasis, "C" => 1.0)),
    (:SVC, Dict("kernel" => LIBSVM.Kernel.Polynomial, "C" => 1.0, "degree" => 2)),
    (:SVC, Dict("kernel" => LIBSVM.Kernel.Polynomial, "C" => 1.0, "degree" => 3)),
    (:SVC, Dict("kernel" => LIBSVM.Kernel.Sigmoid, "C" => 0.5)),


    # Árboles
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

for (modelo, config) in model_configs
    println("\n Evaluando $modelo con parámetros: $(config)")
    acc_tmp = Float64[]

    for fold in 1:k
        println("========== Fold $fold ==========")
        train_idx = findall(folds .!= fold)
        test_idx = findall(folds .== fold)

        df_train = df[train_idx, :]
        df_test = df[test_idx, :]

        # One-hot encoding
        cols_onehot = [:Series, :Surface, :RoundGroup]
        for col in cols_onehot
            clases = unique(df_train[!, col])
            df_train_cod = oneHotEncoding(df_train[!, col], clases, string(col))
            df_test_cod  = oneHotEncoding(df_test[!, col], clases, string(col))
            df_train = hcat(select(df_train, Not(col)), df_train_cod)
            df_test  = hcat(select(df_test, Not(col)), df_test_cod)
        end

        # Normalización
        cols_normalize = [:Pts_Diff_Rel, :Odd_Diff]
        for col in cols_normalize
            μ = mean(df_train[!, col])
            σ = std(df_train[!, col])
            new_name = Symbol("$(col)_normalized")
            if σ ≈ 0
                df_train[!, new_name] .= 0.0
                df_test[!, new_name]  .= 0.0
            else
                df_train[!, new_name] = (df_train[!, col] .- μ) ./ σ
                df_test[!, new_name]  = (df_test[!, col] .- μ) ./ σ
            end
            select!(df_train, Not(col))
            select!(df_test, Not(col))
        end

        # Guardar datasets preprocesados
        CSV.write("train_procesado_fold$fold.csv", df_train)
        CSV.write("test_procesado_fold$fold.csv", df_test)


        # Separar variables
        X_train = Matrix(select(df_train, Not(:Winner)))
        y_train = Vector(df_train.Winner)
        X_test  = Matrix(select(df_test, Not(:Winner)))
        y_test  = Vector(df_test.Winner)

        if modelo == :ANN
            y_train_bin = y_train .== 1
            ann, _ = trainClassANN(config["topology"], (X_train, y_train_bin);
                maxEpochs=100, learningRate=config["learningRate"], minLoss=config["minLoss"])
            y_pred_ann = ann(permutedims(Float32.(X_test)))[:, 1] .> 0.5
            acc = sum(y_pred_ann .== (y_test .== 1)) / length(y_test)
            push!(acc_tmp, acc)

        elseif modelo == :SVC

            if config["kernel"] == "poly"
                model_svm = SVMClassifier(
                    kernel=config["kernel"],
                    cost=config["C"],
                    degree=Int(config["degree"])
                )
            else
                model_svm = SVMClassifier(
                    kernel=config["kernel"],
                    cost=config["C"]
                )
            end
            
            svm_model = machine(model_svm, MLJ.table(X_train), categorical(string.(y_train)))
            MLJ.fit!(svm_model, verbosity=0)
            y_pred_svm = MLJ.predict(svm_model, MLJ.table(X_test))
            acc = sum(y_pred_svm .== string.(y_test)) / length(y_test)
            push!(acc_tmp, acc)

        elseif modelo == :DecisionTreeClassifier
            tree_model = machine(DecisionTreeClassifier(max_depth=config["max_depth"]),
                MLJ.table(X_train), categorical(string.(y_train)))
            MLJ.fit!(tree_model, verbosity=0)
            y_pred_tree = mode.(MLJ.predict(tree_model, MLJ.table(X_test)))
            acc = sum(y_pred_tree .== string.(y_test)) / length(y_test)
            push!(acc_tmp, acc)

        elseif modelo == :KNeighborsClassifier
            knn_model = machine(KNNClassifier(K=config["n_neighbors"]),
                MLJ.table(X_train), categorical(string.(y_train)))
            MLJ.fit!(knn_model, verbosity=0)
            y_pred_knn = mode.(MLJ.predict(knn_model, MLJ.table(X_test)))
            acc = sum(y_pred_knn .== string.(y_test)) / length(y_test)
            push!(acc_tmp, acc)

        elseif modelo == :DoME
            y_pred_dome = trainClassDoME((X_train, y_train .== 1), X_test, config["maximumNodes"])
            y_pred_dome_bin = y_pred_dome .> 0.0
            acc = sum(y_pred_dome_bin .== (y_test .== 1)) / length(y_test)
            push!(acc_tmp, acc)         
        end
    end

    # Guardar resultados de esta configuración
    push!(resultados_configuraciones, (
        string(modelo),
        repr(config),
        mean(acc_tmp),
        std(acc_tmp)
    ))
end

# ========================================
# GUARDAR TODAS LAS CONFIGURACIONES PROBADAS
# ========================================
CSV.write("resultados_todas_configuraciones.csv", resultados_configuraciones)
println("\nGuardado 'resultados_todas_configuraciones.csv' con todas las configuraciones evaluadas.")

# ========================================
# MEJOR CONFIGURACIÓN POR MODELO
# ========================================
mejores_configuraciones = combine(groupby(resultados_configuraciones, :Modelo)) do subdf
    idx = argmax(subdf.AccuracyMedia)
    subdf[idx, :]
end

CSV.write("mejores_configuraciones.csv", mejores_configuraciones)
println("Guardado 'mejores_configuraciones.csv' con la mejor configuración por modelo.")
