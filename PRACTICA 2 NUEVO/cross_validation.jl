using CSV, DataFrames, Statistics, Random
using MLJ, LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface

include("preprocesado.jl")  # Carga `df::DataFrame` y `jugador_dict`
include("modelUtils.jl")    # oneHotEncoding, normalizeZeroMean!, trainClassANN, trainClassDoME, crossvalidation, etc.

# ----------------------------------------
# Función para procesar un fold (train o test)
function procesar_fold(df_fold::DataFrame; one_hot::Bool=false)
    # Copia para no mutar el original
    dfc = deepcopy(df_fold)

    # Separa target (Vector{Bool}) y features
    y = dfc.Winner .== 1
    X = select(dfc, Not(:Winner))

    # 1) One-hot en train y test si pide ANN
    if one_hot
        for col in (:Series, :Surface, :RoundGroup)
            clases = unique(X[!, col])
            enc = oneHotEncoding(X[!, col], clases)  # n×k matrix
            # Añade columnas Series_clase, Surface_clase, …
            for i in 1:size(enc,2)
                X[!, Symbol(col, "_", clases[i])] = enc[:, i]
            end
            select!(X, Not(col))
        end
    end

    # 2) Normalización z-score train→test
    for col in (:Pts_Diff_Rel, :Odd_Diff)
        v = X[!, col]
        μ = mean(v)
        σ = std(v)
        X[!, col] = σ == 0 ? zeros(length(v)) : (v .- μ) ./ σ
    end

    return Matrix(X), y
end

# ----------------------------------------
# Configuración cross-validation
k = 5
folds = crossvalidation(df.Winner, k)

# Preparar DataFrame de resultados
resultados = DataFrame(
  Modelo       = String[],
  Parametros   = String[],
  AccuracyMedia = Float64[],
  AccuracyStd  = Float64[]
)

# Modelos a evaluar
model_configs = [
  (:ANN,                      Dict("topology"=>[16], "learningRate"=>0.01, "minLoss"=>0.005)),
  (:SVC,                      Dict("kernel"=>"linear", "C"=>1.0)),
  (:DecisionTreeClassifier,   Dict("max_depth"=>3)),
  (:KNeighborsClassifier,     Dict("n_neighbors"=>5)),
  (:DoME,                     Dict("maximumNodes"=>4))
]

# ----------------------------------------
# Bucle sobre modelos y folds
for (modelo, config) in model_configs
  accuracies = Float64[]
  for fold in 1:k
    # separa índices
    train_idx = findall(folds .!= fold)
    test_idx  = findall(folds .== fold)
    df_train = df[train_idx, :]
    df_test  = df[test_idx, :]

    # preprocesado
    one_hot = (modelo == :ANN)
    X_train, y_train = procesar_fold(df_train; one_hot=one_hot)
    X_test,  y_test  = procesar_fold(df_test;  one_hot=one_hot)

    # entrenamiento + predicción
    if modelo == :ANN
      ann, _ = trainClassANN(
        config["topology"],
        (X_train, y_train);
        learningRate = config["learningRate"],
        minLoss      = config["minLoss"]
      )
      y_pred = vec(classifyOutputs(ann(Float32.(X_test'))'))
      y_true = y_test

    elseif modelo == :DoME
      y_pred = trainClassDoME((X_train, y_train), X_test, config["maximumNodes"])
      y_true = y_test

    else
      # SVC, DecisionTree o KNN vía MLJ
      model = modelo == :SVC ? SVC(kernel=LIBSVM.Kernel.Linear, cost=config["C"]) :
              modelo == :DecisionTreeClassifier ?
                (@load DecisionTreeClassifier pkg=DecisionTree verbosity=0)(max_depth=config["max_depth"]) :
              modelo == :KNeighborsClassifier ?
                (@load KNNClassifier pkg=NearestNeighborModels verbosity=0)(K=config["n_neighbors"]) :
              error("Modelo desconocido")
      mach = machine(model, MLJ.table(X_train), categorical(string.(y_train)))
      fit!(mach, verbosity=0)
      y_pred = mode.(predict(mach, MLJ.table(X_test)))
      y_true = string.(y_test)
    end

    push!(accuracies, mean(y_pred .== y_true))
  end

  # almacena resultados
  push!(resultados, (
    string(modelo),
    repr(config),
    round(mean(accuracies), digits=4),
    round(std(accuracies), digits=4)
  ))
end

# ----------------------------------------
# Guardar y mostrar
CSV.write("resultados.csv", resultados)
println("✅ Resultados guardados en resultados.csv")
