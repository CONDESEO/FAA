# run_experiment.jl

#####################################################
# 1. CARGA DE PAQUETES Y DE TUS FUNCIONES PREVIAS
#####################################################
using CSV, DataFrames, Random
using MLJ                                   # API común de MLJ
using MLJLIBSVMInterface                   # para MLJLIBSVMInterface.SVC
using MLJDecisionTreeInterface             # para DTClassifier
using NearestNeighborModels                # para kNNClassifier

include("funciones.jl")                    # utilidades: preprocesado, modelos, métricas
include("preprocessing.jl")                # carga `inputs` y `targets`

#####################################################
# 2. FIJAR SEMILLA PARA REPRODUCIBILIDAD
#####################################################
seed!(1234)

#####################################################
# 3. LEER ÍNDICES DE VALIDACIÓN CRUZADA
#####################################################
df_idx = CSV.read("indices_cv5.csv", DataFrame)
folds  = df_idx.fold

#####################################################
# 4. DEFINIR MODELOS Y SUS HIPERPARÁMETROS
#####################################################
modelos = Dict(
  :ANN  => [ Dict("topology"=>[16,8],  "learningRate"=>0.01),
             Dict("topology"=>[32,16], "learningRate"=>0.005) ],
  :SVC  => [ Dict("kernel"=>"rbf",    "C"=>1.0),
             Dict("kernel"=>"linear", "C"=>0.1) ],
  :DecisionTree => [ Dict("max_depth"=>5),  Dict("max_depth"=>10) ],
  :kNN  => [ Dict("n_neighbors"=>3), Dict("n_neighbors"=>5) ],
  :DoME => [ Dict("maximumNodes"=>5), Dict("maximumNodes"=>10) ]
)

#####################################################
# 5. PREPARAR ESTRUCTURA PARA ALMACENAR RESULTADOS
#####################################################
resultados = DataFrame(
  Model    = String[],
  Params   = String[],
  MeanAcc  = Float64[],
  StdAcc   = Float64[]
)

#####################################################
# 6. BUCLE PRINCIPAL DE EXPERIMENTOS
#####################################################
for (modelo, configs) in modelos
  for params in configs
    accs = Float64[]

    # 6.1 Validación cruzada en 5 folds
    for f in 1:5
      train_mask = folds .!= f
      test_mask  = folds .== f

      Xtr, ytr = inputs[train_mask, :], targets[train_mask, :]
      Xte, yte = inputs[test_mask,  :], targets[test_mask, :]

      # 6.2 Entrenar y predecir
      if modelo == :ANN
        ann, _ = trainClassANN(
                  params["topology"],
                  (Xtr, ytr);
                  learningRate = params["learningRate"]
                )
        preds = collect(ann(Float32.(Xte'))')

      elseif modelo == :SVC
        svm = MLJLIBSVMInterface.SVC(
                kernel = params["kernel"] == "rbf"    ? LIBSVM.Kernel.RadialBasis :
                         params["kernel"] == "linear" ? LIBSVM.Kernel.Linear  : nothing,
                cost   = params["C"]
              )
        mach = machine(svm, MLJ.table(Xtr), categorical(ytr))
        fit!(mach, verbosity=0)
        preds = mode.(predict(mach, MLJ.table(Xte)))

      elseif modelo == :DecisionTree
        tree = DTClassifier(max_depth=params["max_depth"])
        mach = machine(tree, MLJ.table(Xtr), categorical(ytr))
        fit!(mach, verbosity=0)
        preds = mode.(predict(mach, MLJ.table(Xte)))

      elseif modelo == :kNN
        knn = kNNClassifier(K=params["n_neighbors"])
        mach = machine(knn, MLJ.table(Xtr), categorical(ytr))
        fit!(mach, verbosity=0)
        preds = mode.(predict(mach, MLJ.table(Xte)))

      elseif modelo == :DoME
        preds = trainClassDoME((Xtr, ytr), Xte, params["maximumNodes"])

      else
        error("Modelo desconocido: $modelo")
      end

      # 6.3 Almacenar la accuracy de este fold
      push!(accs, accuracy(preds, yte))
    end

    # 6.4 Calcular estadísticas y guardar
    media = mean(accs)
    desvi = std(accs)
    push!(resultados, (
      string(modelo),
      repr(params),
      media,
      desvi
    ))
  end
end

#####################################################
# 7. GUARDAR RESULTADOS EN CSV
#####################################################
CSV.write("resultados.csv", resultados)
println("✅ Experimentos completados. Resultados en resultados.csv")
