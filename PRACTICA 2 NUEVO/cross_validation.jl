using CSV, DataFrames, Statistics, Random
import Statistics: std
using MLJ, LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface
using Flux
using Flux.Losses



include("modelUtils.jl")
include("preprocesado.jl")

# folds de validación cruzada estratificada
folds = crossvalidation(df.Winner, 5)
k = 5

# Diccionarios para guardar los resultados de cada modelo por fold
acc_ann = Float64[]
acc_svm = Float64[]
acc_tree = Float64[]
acc_knn = Float64[]
acc_dome = Float64[]

for fold in 1:k
    println("\n=========== Fold $fold ===========")

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

        println("Codificación One-hot $col exitosa")
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

        println("Normalización Zero-mean $col exitosa")
    end

    # Guardar dataset procesado por fold (opcional)
    df_fold = vcat(df_train, df_test)
    CSV.write("atp_tennis_procesado_fold$fold.csv", df_fold)

    # Separar variables
    X_train = Matrix(select(df_train, Not(:Winner)))
    y_train = Vector(df_train.Winner)

    y_train_bin = y_train .== 1

    X_test = Matrix(select(df_test, Not(:Winner)))
    y_test = Vector(df_test.Winner)

    # Entrenamiento de modelos

    # === ANN ===
    ann, _ = trainClassANN([10], (X_train, y_train_bin); maxEpochs=50, learningRate=0.01)
    y_pred_ann = ann(Float32.(X_test'))'[:, 1] .> 0.5
    acc = sum(y_pred_ann .== y_test) / length(y_test)
    push!(acc_ann, acc)
    println("Accuracy ANN: ", acc)

    # === SVM ===
    modelo = SVMClassifier()
    svm_model = machine(modelo, MLJ.table(X_train), categorical(string.(y_train)))
    MLJ.fit!(svm_model, verbosity=0)
    y_pred_svm = MLJ.predict(svm_model, MLJ.table(X_test))
    acc = sum(y_pred_svm .== string.(y_test)) / length(y_test)
    push!(acc_svm, acc)
    println("Accuracy SVM: ", acc)

    # === Árbol ===
    tree_model = machine(DecisionTreeClassifier(max_depth=5), MLJ.table(X_train), categorical(string.(y_train)))
    MLJ.fit!(tree_model, verbosity=0)
    y_pred_tree = mode.(MLJ.predict(tree_model, MLJ.table(X_test)))
    acc = sum(y_pred_tree .== string.(y_test)) / length(y_test)
    push!(acc_tree, acc)
    println("Accuracy Árbol: ", acc)

    # === kNN ===
    knn_model = machine(KNNClassifier(K=5), MLJ.table(X_train), categorical(string.(y_train)))
    MLJ.fit!(knn_model, verbosity=0)
    y_pred_knn = MLJ.predict(knn_model, MLJ.table(X_test))
    acc = sum(y_pred_knn .== string.(y_test)) / length(y_test)
    push!(acc_knn, acc)
    println("Accuracy kNN: ", acc)

    # === DoME ===
    y_train_dome = y_train .== 1
    y_test_dome = string.(y_test)
    println(typeof(y_train_dome))
    y_pred_dome = trainClassDoME((X_train, y_train_dome), X_test, 3)
    acc = sum(y_pred_dome .== y_test_dome) / length(y_test_dome)
    push!(acc_dome, acc)
    println("Accuracy DoME: ", acc)
end

# =========================
# GUARDAR RESULTADOS FINALES
# =========================

resultados = DataFrame(
    Modelo = ["ANN", "SVM", "Árbol", "kNN", "DoME"],
    Accuracy = [mean(acc_ann), mean(acc_svm), mean(acc_tree), mean(acc_knn), mean(acc_dome)],
    Desviacion = [std(acc_ann), std(acc_svm), std(acc_tree), std(acc_knn), std(acc_dome)]
)

CSV.write("resultados_finales.csv", resultados)
println("\n✅ Resultados guardados en 'resultados_finales.csv'")
