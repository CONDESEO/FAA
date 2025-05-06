# -------------------------------------------------------------------------
# Paquetes requeridos para todas las funciones
# -------------------------------------------------------------------------

using CUDA
using CSV, DataFrames, Random
using Statistics           # Para funciones estadísticas (mean, std, etc.)
using StatsBase
using Flux                 # Para construir y entrenar redes neuronales
using Flux.Losses          # Para funciones de pérdida como crossentropy
using Random               # Para aleatoriedad en Hold-Out y CV
using Random: seed!        # Para inicializar semilla si se desea
using SymDoME              # Para entrenar modelos DoME
using MLJ                  # Para SVM, kNN, árboles...
using LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface

# -------------------------------------------------------------------------
# Funciones para codificar entradas y salidas categóricas
# -------------------------------------------------------------------------

# Codifica en one-hot con clases especificadas
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    @assert(all([in(value, classes) for value in feature]))
    numClasses = length(classes)
    if numClasses <= 2
        return reshape(feature .== classes[1], :, 1)
    else
        return convert(BitArray{2}, hcat([instance .== classes for instance in feature]...)')
    end
end

# Codifica en one-hot tomando clases automáticamente
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

# Si ya es booleano, lo convierte a columna
oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1)



# -------------------------------------------------------------------------
# FUNCIONES DE NORMALIZACIÓN
# -------------------------------------------------------------------------

# Calculan los parámetros de normalización
calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2}) =
    (minimum(dataset, dims=1), maximum(dataset, dims=1))

calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2}) =
    (mean(dataset, dims=1), std(dataset, dims=1))

# Normalización Min-Max (con modificación del dataset original)
function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues, maxValues = normalizationParameters
    dataset .-= minValues
    dataset ./= (maxValues .- minValues)
    dataset[:, vec(minValues .== maxValues)] .= 0
    return dataset
end

normalizeMinMax!(dataset::AbstractArray{<:Real,2}) =
    normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset))

normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) =
    normalizeMinMax!(copy(dataset), normalizationParameters)

normalizeMinMax(dataset::AbstractArray{<:Real,2}) =
    normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset))

# Normalización media 0 y desviación típica 1 (con modificación del dataset original)
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    avgValues, stdValues = normalizationParameters
    dataset .-= avgValues
    dataset ./= stdValues
    dataset[:, vec(stdValues .== 0)] .= 0
    return dataset
end

normalizeZeroMean!(dataset::AbstractArray{<:Real,2}) =
    normalizeZeroMean!(dataset, calculateZeroMeanNormalizationParameters(dataset))

normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) =
    normalizeZeroMean!(copy(dataset), normalizationParameters)

normalizeZeroMean(dataset::AbstractArray{<:Real,2}) =
    normalizeZeroMean!(copy(dataset), calculateZeroMeanNormalizationParameters(dataset))



# -------------------------------------------------------------------------
# FUNCIONES DE CLASIFICACIÓN DE SALIDAS
# -------------------------------------------------------------------------

# Clasifica una salida real como booleana con umbral
classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5) = outputs .>= threshold

# Clasifica una matriz de salidas, eligiendo la clase con mayor probabilidad
function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    numOutputs = size(outputs, 2)
    @assert(numOutputs != 2)
    if numOutputs == 1
        return reshape(classifyOutputs(outputs[:]; threshold=threshold), :, 1)
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2)
        classified = falses(size(outputs))
        classified[indicesMaxEachInstance] .= true
        @assert(all(sum(classified, dims=2) .== 1))
        return classified
    end
end

# -------------------------------------------------------------------------
# FUNCIONES DE PRECISIÓN
# -------------------------------------------------------------------------

# Precisión para vectores booleanos
accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) = mean(outputs .== targets)

# Precisión para matrices booleanas
function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    @assert(all(size(outputs) .== size(targets)))
    if size(targets, 2) == 1
        return accuracy(outputs[:,1], targets[:,1])
    else
        return mean(all(targets .== outputs, dims=2))
    end
end

# Precisión para vectores reales (con umbral)
accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) =
    accuracy(outputs .>= threshold, targets)

# Precisión para matrices reales (con umbral)
function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    @assert(all(size(outputs) .== size(targets)))
    if size(targets, 2) == 1
        return accuracy(outputs[:,1], targets[:,1]; threshold=threshold)
    else
        return accuracy(classifyOutputs(outputs), targets)
    end
end


# -------------------------------------------------------------------------
# FUNCIONES PARA CREAR Y ENTRENAR UNA RNA
# -------------------------------------------------------------------------

# Crea una red neuronal según la topología y funciones de activación
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
ann = Chain()
numInputsLayer = numInputs
for numHiddenLayer in 1:length(topology)
numNeurons = topology[numHiddenLayer]
ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]))
numInputsLayer = numNeurons
end
if numOutputs == 1
ann = Chain(ann..., Dense(numInputsLayer, 1, σ))
else
ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity), softmax)
end
return ann
end

# Entrena una RNA con entrenamiento simple
function trainClassANN(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    inputs, targets = dataset
    @assert size(inputs,1) == size(targets,1)

    inputs = gpu(Float32.(inputs))
    targets = gpu(Float32.(targets))

    ann = buildClassANN(size(inputs,2), topology, size(targets,2), transferFunctions=transferFunctions)
    ann = gpu(ann)  # ⚠️ Esto debe ir ANTES de calcular la loss

    loss(model, x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)
    
    trainingLosses = Float32[]
    numEpoch = 0
    trainingLoss = loss(ann, inputs', targets')
    push!(trainingLosses, Float32(cpu(trainingLoss)))  # CPU para poder imprimir/guardar
    
    println("Epoch ", numEpoch, ": loss: ", trainingLoss)

    opt_state = Flux.setup(Adam(learningRate), ann)

    while (numEpoch < maxEpochs) && (trainingLoss > minLoss)
        Flux.train!(loss, ann, [(inputs', targets')], opt_state)
        numEpoch += 1
        trainingLoss = loss(ann, inputs', targets')
        push!(trainingLosses, Float32(cpu(trainingLoss)))  # CPU para almacenar
        println("Epoch ", numEpoch, ": loss: ", trainingLoss)
    end

    return (ann, trainingLosses)
end

# Versión para vector
trainClassANN(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01) =
    trainClassANN(topology, (dataset[1], reshape(dataset[2], :, 1));
        transferFunctions=transferFunctions,
        maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate)



# -------------------------------------------------------------------------
# FUNCIONES DE DIVISIÓN HOLD-OUT Y ENTRENAMIENTO CON VALIDACIÓN/TEST
# -------------------------------------------------------------------------

# División en conjunto de entrenamiento y test (hold-out)
function holdOut(N::Int, P::Real)
    @assert (0.0 <= P <= 1.0)
    indices = randperm(N)
    numTraining = Int(round(N * (1 - P)))
    return (indices[1:numTraining], indices[numTraining+1:end])
end

# División en entrenamiento, validación y test (hold-out extendido)
function holdOut(N::Int, Pval::Real, Ptest::Real)
    @assert (0.0 <= Pval <= 1.0)
    @assert (0.0 <= Ptest <= 1.0)
    @assert (Pval + Ptest <= 1.0)
    (trainValIndices, testIndices) = holdOut(N, Ptest)
    (trainIndices, valIndices) = holdOut(length(trainValIndices), Pval * N / length(trainValIndices))
    return (trainValIndices[trainIndices], trainValIndices[valIndices], testIndices)
end

# Entrena una RNA con posibilidad de validación y test
function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(
        Array{eltype(trainingDataset[1]),2}(undef, 0, size(trainingDataset[1],2)),
        falses(0, size(trainingDataset[2],2))),
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(
        Array{eltype(trainingDataset[1]),2}(undef, 0, size(trainingDataset[1],2)),
        falses(0, size(trainingDataset[2],2))),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)

# Datos originales
trainingInputs, trainingTargets = trainingDataset
validationInputs, validationTargets = validationDataset
testInputs, testTargets = testDataset

# Comprobaciones de tamaño
@assert size(trainingInputs,1) == size(trainingTargets,1)
@assert size(validationInputs,1) == size(validationTargets,1)
@assert size(testInputs,1) == size(testTargets,1)

# Normalización de dimensiones
!isempty(validationInputs) && @assert size(trainingInputs,2) == size(validationInputs,2)
!isempty(testInputs)       && @assert size(trainingInputs,2) == size(testInputs,2)
!isempty(validationTargets) && @assert size(trainingTargets,2) == size(validationTargets,2)
!isempty(testTargets)       && @assert size(trainingTargets,2) == size(testTargets,2)

# PASAR TODO A GPU
trainingInputs = gpu(Float32.(trainingInputs))
trainingTargets = gpu(Float32.(trainingTargets))
validationInputs = gpu(Float32.(validationInputs))
validationTargets = gpu(Float32.(validationTargets))
testInputs = gpu(Float32.(testInputs))
testTargets = gpu(Float32.(testTargets))

# Modelo en GPU
ann = buildClassANN(size(trainingInputs,2), topology, size(trainingTargets,2); transferFunctions=transferFunctions)
ann = gpu(ann)

# Función de pérdida
loss(model, x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)

trainingLosses = Float32[]
validationLosses = Float32[]
testLosses = Float32[]
numEpoch = 0

function calculateLossValues()
trainLoss = loss(ann, trainingInputs', trainingTargets')
valLoss = isempty(validationInputs) ? NaN : loss(ann, validationInputs', validationTargets')
testLoss = isempty(testInputs) ? NaN : loss(ann, testInputs', testTargets')
push!(trainingLosses, Float32(cpu(trainLoss)))
!isempty(validationInputs) && push!(validationLosses, Float32(cpu(valLoss)))
!isempty(testInputs) && push!(testLosses, Float32(cpu(testLoss)))
return (trainLoss, valLoss, testLoss)
end

(trainingLoss, validationLoss, _) = calculateLossValues()
if isempty(validationInputs); maxEpochsVal = Inf; end

numEpochsVal = 0
bestValLoss = validationLoss
bestANN = deepcopy(ann)
opt_state = Flux.setup(Adam(learningRate), ann)

while (numEpoch < maxEpochs) && (trainingLoss > minLoss) && (numEpochsVal < maxEpochsVal)
Flux.train!(loss, ann, [(trainingInputs', trainingTargets')], opt_state)
numEpoch += 1
(trainingLoss, validationLoss, _) = calculateLossValues()
println("Epoch $numEpoch | Training Loss: $(Float32(cpu(trainingLoss)))")
if !isempty(validationInputs)
if validationLoss < bestValLoss
bestValLoss = validationLoss
numEpochsVal = 0
bestANN = deepcopy(ann)
else
numEpochsVal += 1
end
end
end

if isempty(validationInputs)
bestANN = ann
end

return (bestANN, trainingLosses, validationLosses, testLosses)
end


# -------------------------------------------------------------------------
# FUNCIONES DE EVALUACIÓN: Accuracy, Confusion Matrix y Visualización
# -------------------------------------------------------------------------

# Accuracy para salidas booleanas
accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) = mean(outputs .== targets)

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    @assert size(outputs) == size(targets)
    if size(targets,2) == 1
        return accuracy(outputs[:,1], targets[:,1])
    else
        return mean(all(targets .== outputs, dims=2))
    end
end

# Accuracy para salidas reales (umbral)
accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) =
    accuracy(outputs .>= threshold, targets)

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    @assert size(outputs) == size(targets)
    if size(targets,2) == 1
        return accuracy(outputs[:,1], targets[:,1]; threshold=threshold)
    else
        return accuracy(classifyOutputs(outputs), targets)
    end
end

# Devuelve una matriz booleana indicando la clase más probable
classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5) = outputs .>= threshold

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    numOutputs = size(outputs,2)
    @assert(numOutputs != 2)
    if numOutputs == 1
        return reshape(classifyOutputs(outputs[:]; threshold=threshold), :, 1)
    else
        (_, maxIndices) = findmax(outputs, dims=2)
        result = falses(size(outputs))
        result[maxIndices] .= true
        @assert all(sum(result, dims=2) .== 1)
        return result
    end
end

# Matriz de confusión binaria
function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    TN = sum(.!outputs .& .!targets)
    FN = sum(.!outputs .& targets)
    TP = sum(outputs .& targets)
    FP = sum(outputs .& .!targets)

    acc = (TP + TN) / (TP + TN + FP + FN)
    errorRate = 1 - acc
    recall = (TP == FN == 0) ? 1. : TP / (TP + FN)
    specificity = (TN == FP == 0) ? 1. : TN / (TN + FP)
    precision = (TP == FP == 0) ? 1. : TP / (TP + FP)
    NPV = (TN == FN == 0) ? 1. : TN / (TN + FN)
    F1 = (recall == precision == 0) ? 0. : 2 * (recall * precision) / (recall + precision)
    @assert !isnan(F1)

    confMatrix = [TN FP;
                  FN TP]
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end

confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) =
    confusionMatrix(outputs .>= threshold, targets)

# Matriz de confusión multiclase
function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert size(outputs) == size(targets)
    (N, C) = size(targets)
    @assert C != 2
    if C == 1
        return confusionMatrix(outputs[:,1], targets[:,1])
    end
    @assert all(sum(outputs, dims=2) .== 1)

    recall = zeros(C)
    specificity = zeros(C)
    precision = zeros(C)
    NPV = zeros(C)
    F1 = zeros(C)

    for i in 1:C
        (_, _, recall[i], specificity[i], precision[i], NPV[i], F1[i], _) =
            confusionMatrix(outputs[:,i], targets[:,i])
    end

    confMatrix = [sum(targets[:,i] .& outputs[:,j]) for i in 1:C, j in 1:C]

    if weighted
        classCounts = vec(sum(targets, dims=1))
        weights = classCounts ./ sum(classCounts)
        recall = sum(weights .* recall)
        specificity = sum(weights .* specificity)
        precision = sum(weights .* precision)
        NPV = sum(weights .* NPV)
        F1 = sum(weights .* F1)
    else
        recall = mean(recall)
        specificity = mean(specificity)
        precision = mean(precision)
        NPV = mean(NPV)
        F1 = mean(F1)
    end

    acc = accuracy(outputs, targets)
    errorRate = 1 - acc
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end

confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true) =
    confusionMatrix(classifyOutputs(outputs; threshold=threshold), targets; weighted=weighted)

# Matriz de confusión multiclase para etiquetas categóricas
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert all([in(label, classes) for label in vcat(outputs, targets)])
    return confusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted)
end

confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true) =
    confusionMatrix(outputs, targets, unique(vcat(outputs, targets)); weighted=weighted)

# Imprime la matriz de confusión por pantalla
function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix = confusionMatrix(outputs, targets; weighted=weighted)
    C = size(confMatrix, 1)
    println("Confusion Matrix:")
    println("\t| ", join(["Cl. $i" for i in 1:C], "\t| "), "\t|")
    println("-"^8^C)
    for i in 1:C
        print("Cl. $i\t| ")
        print(join(confMatrix[i,:], "\t| "), "\t|\n")
    end
    println("-"^8^C)
    println("Accuracy: ", acc)
    println("Error rate: ", errorRate)
    println("Recall: ", recall)
    println("Specificity: ", specificity)
    println("Precision: ", precision)
    println("Negative Predictive Value: ", NPV)
    println("F1-score: ", F1)
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end

# Sobrecargas de printConfusionMatrix para otros tipos
printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) =
    printConfusionMatrix(classifyOutputs(outputs), targets; weighted=weighted)

printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) =
    printConfusionMatrix(reshape(outputs, :, 1), reshape(targets, :, 1))

printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) =
    printConfusionMatrix(outputs .>= threshold, targets)

printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true) =
    printConfusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted)

function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs))
    printConfusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted)
end



# -------------------------------------------------------------------------
# ENTRENAMIENTO CON DoME (Símbolo de Ecuaciones)
# -------------------------------------------------------------------------

# Entrenamiento para etiquetas booleanas (vector columna)
function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
    testInputs::AbstractArray{<:Real,2},
    maximumNodes::Int)
(trainingInputs, trainingTargets) = trainingDataset
trainingInputs = Float64.(trainingInputs)
testInputs = Float64.(testInputs)

_, _, _, model = dome(trainingInputs, trainingTargets; maximumNodes=maximumNodes)

testOutputs = evaluateTree(model, testInputs)
if isa(testOutputs, Real)
testOutputs = repeat([testOutputs], size(testInputs, 1))
end
return testOutputs
end

# Entrenamiento para etiquetas booleanas codificadas (matriz)
function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
    testInputs::AbstractArray{<:Real,2},
    maximumNodes::Int)
(trainingInputs, trainingTargets) = trainingDataset
numClasses = size(trainingTargets, 2)

if numClasses == 1
testOutputs = trainClassDoME((trainingInputs, vec(trainingTargets)), testInputs, maximumNodes)
return reshape(testOutputs, :, 1)
end

@assert numClasses > 2

testOutputs = Array{Float64,2}(undef, size(testInputs,1), numClasses)
for i in 1:numClasses
testOutputs[:, i] .= trainClassDoME((trainingInputs, trainingTargets[:, i]), testInputs, maximumNodes)
end
return testOutputs
end

# Entrenamiento para etiquetas categóricas (no codificadas)
function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    testInputs::AbstractArray{<:Real,2},
    maximumNodes::Int)
(trainingInputs, trainingTargets) = trainingDataset
classes = unique(trainingTargets)

testOutputsDoME = trainClassDoME((trainingInputs, oneHotEncoding(trainingTargets, classes)), testInputs, maximumNodes)
testOutputsBool = classifyOutputs(testOutputsDoME; threshold=0)

testOutputs = Array{eltype(trainingTargets),1}(undef, size(testInputs,1))

if length(classes) <= 2
@assert isa(testOutputsBool, Vector) || size(testOutputsBool,2) == 1
testOutputs[testOutputsBool] .= classes[1]
if length(classes) == 2
testOutputs[.!testOutputsBool] .= classes[2]
else
@assert all(testOutputsBool)
end
else
@assert all(sum(testOutputsBool, dims=2) .== 1)
for i in eachindex(classes)
testOutputs[testOutputsBool[:, i]] .= classes[i]
end
end

return testOutputs
end



# -------------------------------------------------------------------------
# VALIDACIÓN CRUZADA Y HOLD-OUT
# -------------------------------------------------------------------------

# Genera conjuntos train/test con hold-out
function holdOut(N::Int, P::Real)
    @assert 0.0 <= P <= 1.0
    indices = randperm(N)
    numTrain = Int(round(N * (1 - P)))
    return (indices[1:numTrain], indices[numTrain+1:end])
end

# Hold-out para train/val/test
function holdOut(N::Int, Pval::Real, Ptest::Real)
    @assert 0.0 <= Pval <= 1.0
    @assert 0.0 <= Ptest <= 1.0
    @assert Pval + Ptest <= 1.0
    (trainVal, test) = holdOut(N, Ptest)
    (train, val) = holdOut(length(trainVal), Pval * N / length(trainVal))
    return (trainVal[train], trainVal[val], test)
end

# Validación cruzada estratificada para clasificación binaria o multiclase
function crossvalidation(targets::AbstractArray{Bool,1}, k::Int)
    indices = Array{Int64,1}(undef, length(targets))
    indices[targets] = crossvalidation(sum(targets), k)
    indices[.!targets] = crossvalidation(sum(.!targets), k)
    return indices
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int)
    numClasses = size(targets, 2)
    @assert numClasses != 2
    if numClasses == 1
        return crossvalidation(vec(targets), k)
    end
    indices = Array{Int64,1}(undef, size(targets, 1))
    for i in 1:numClasses
        indices[targets[:, i]] = crossvalidation(sum(targets[:, i]), k)
    end
    return indices
end

crossvalidation(N::Int, k::Int) = shuffle!(repeat(1:k, ceil(Int, N/k))[1:N])
crossvalidation(targets::AbstractArray{<:Any,1}, k::Int) = crossvalidation(oneHotEncoding(targets), k)


# -------------------------------------------------------------------------
# VALIDACIÓN CRUZADA PARA RNA (ANN)
# -------------------------------------------------------------------------

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    validationRatio::Real=0, maxEpochsVal::Int=20)

    (inputs, targets) = dataset
    @assert size(inputs,1) == length(targets)

    classes = unique(targets)
    targets = oneHotEncoding(targets, classes)
    numFolds = maximum(crossValidationIndices)

    testAccuracy = zeros(Float64, numFolds)
    testErrorRate = zeros(Float64, numFolds)
    testRecall = zeros(Float64, numFolds)
    testSpecificity = zeros(Float64, numFolds)
    testPrecision = zeros(Float64, numFolds)
    testNPV = zeros(Float64, numFolds)
    testF1 = zeros(Float64, numFolds)
    testConfMatrix = zeros(length(classes), length(classes))

    for fold in 1:numFolds
        trainX = inputs[crossValidationIndices .!= fold, :]
        testX = inputs[crossValidationIndices .== fold, :]
        trainY = targets[crossValidationIndices .!= fold, :]
        testY = targets[crossValidationIndices .== fold, :]

        accs = zeros(Float64, numExecutions)
        errs = zeros(Float64, numExecutions)
        recs = zeros(Float64, numExecutions)
        specs = zeros(Float64, numExecutions)
        precs = zeros(Float64, numExecutions)
        npvs = zeros(Float64, numExecutions)
        f1s = zeros(Float64, numExecutions)
        confs = Array{Float64,3}(undef, length(classes), length(classes), numExecutions)

        for rep in 1:numExecutions
            if validationRatio > 0
                (idxTrain, idxVal) = holdOut(size(trainX,1), validationRatio * size(inputs,1) / size(trainX,1))
                ann, = trainClassANN(topology, (trainX[idxTrain,:], trainY[idxTrain,:]),
                    validationDataset=(trainX[idxVal,:], trainY[idxVal,:]),
                    testDataset=(testX, testY),
                    transferFunctions=transferFunctions,
                    maxEpochs=maxEpochs, minLoss=minLoss,
                    learningRate=learningRate, maxEpochsVal=maxEpochsVal)
            else
                ann, = trainClassANN(topology, (trainX, trainY),
                    testDataset=(testX, testY),
                    transferFunctions=transferFunctions,
                    maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate)
            end

            preds = collect(cpu(ann(gpu(Float32.(testX')))))

            accs[rep], errs[rep], recs[rep], specs[rep], precs[rep], npvs[rep], f1s[rep], confs[:,:,rep] =
                confusionMatrix(preds', testY)
        end

        testAccuracy[fold]    = mean(accs)
        testErrorRate[fold]   = mean(errs)
        testRecall[fold]      = mean(recs)
        testSpecificity[fold] = mean(specs)
        testPrecision[fold]   = mean(precs)
        testNPV[fold]         = mean(npvs)
        testF1[fold]          = mean(f1s)
        testConfMatrix       += mean(confs, dims=3)[:,:,1]
    end

    return (mean(testAccuracy), std(testAccuracy)),
           (mean(testErrorRate), std(testErrorRate)),
           (mean(testRecall), std(testRecall)),
           (mean(testSpecificity), std(testSpecificity)),
           (mean(testPrecision), std(testPrecision)),
           (mean(testNPV), std(testNPV)),
           (mean(testF1), std(testF1)),
           testConfMatrix
end


# -------------------------------------------------------------------------
# VALIDACIÓN CRUZADA GENERAL PARA DIFERENTES MODELOS
# -------------------------------------------------------------------------

function modelCrossValidation(
    modelType::Symbol,
    modelHyperparameters::Dict,
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1}
)

    (inputs, targets) = dataset
    @assert size(inputs,1) == length(targets)

    if modelType == :ANN
        return ANNCrossValidation(
            modelHyperparameters["topology"],
            dataset, crossValidationIndices;
            numExecutions     = get(modelHyperparameters, "numExecutions", 50),
            transferFunctions = get(modelHyperparameters, "transferFunctions", fill(σ, length(modelHyperparameters["topology"]))),
            maxEpochs         = get(modelHyperparameters, "maxEpochs", 1000),
            minLoss           = get(modelHyperparameters, "minLoss", 0.0),
            learningRate      = get(modelHyperparameters, "learningRate", 0.01),
            validationRatio   = get(modelHyperparameters, "validationRatio", 0),
            maxEpochsVal      = get(modelHyperparameters, "maxEpochsVal", 20)
        )
    end

    # Si no es ANN, pasamos las salidas a string
    targets = string.(targets)
    classes = unique(targets)
    numFolds = maximum(crossValidationIndices)

    testAccuracy        = zeros(Float64, numFolds)
    testErrorRate       = zeros(Float64, numFolds)
    testRecall          = zeros(Float64, numFolds)
    testSpecificity     = zeros(Float64, numFolds)
    testPrecision       = zeros(Float64, numFolds)
    testNPV             = zeros(Float64, numFolds)
    testF1              = zeros(Float64, numFolds)
    testConfMatrix      = zeros(Int, length(classes), length(classes))

    for fold in 1:numFolds
        trainX = inputs[crossValidationIndices .!= fold, :]
        testX  = inputs[crossValidationIndices .== fold, :]
        trainY = targets[crossValidationIndices .!= fold]
        testY  = targets[crossValidationIndices .== fold]

        if modelType == :DoME
            preds = trainClassDoME((trainX, trainY), testX, modelHyperparameters["maximumNodes"])
        else
            model = if modelType == :SVC
                kernel = get(modelHyperparameters, "kernel", "linear")
                @assert kernel in ["linear", "poly", "rbf", "sigmoid"]
                SVMClassifier(
                    kernel = kernel == "linear"  ? LIBSVM.Kernel.Linear :
                             kernel == "rbf"     ? LIBSVM.Kernel.RadialBasis :
                             kernel == "poly"    ? LIBSVM.Kernel.Polynomial :
                             kernel == "sigmoid" ? LIBSVM.Kernel.Sigmoid : nothing,
                    cost   = Float64(modelHyperparameters["C"]),
                    gamma  = Float64(get(modelHyperparameters, "gamma", -1)),
                    degree = Int32(get(modelHyperparameters, "degree", -1)),
                    coef0  = Float64(get(modelHyperparameters, "coef0", -1))
                )
            elseif modelType == :DecisionTreeClassifier
                DTClassifier(max_depth = modelHyperparameters["max_depth"], rng=Random.MersenneTwister(1))
            elseif modelType == :KNeighborsClassifier
                kNNClassifier(K = modelHyperparameters["n_neighbors"])
            else
                error("Unknown model $modelType")
            end

            mach = machine(model, MLJ.table(trainX), categorical(trainY))
            MLJ.fit!(mach, verbosity=0)
            preds = MLJ.predict(mach, MLJ.table(testX))
            if modelType != :SVC
                preds = mode.(preds)
            end
        end

        acc, err, rec, spec, prec, npv, f1, conf =
            confusionMatrix(preds, testY, classes)

        testAccuracy[fold]    = acc
        testErrorRate[fold]   = err
        testRecall[fold]      = rec
        testSpecificity[fold] = spec
        testPrecision[fold]   = prec
        testNPV[fold]         = npv
        testF1[fold]          = f1
        testConfMatrix       += conf
    end

    return (mean(testAccuracy), std(testAccuracy)),
           (mean(testErrorRate), std(testErrorRate)),
           (mean(testRecall), std(testRecall)),
           (mean(testSpecificity), std(testSpecificity)),
           (mean(testPrecision), std(testPrecision)),
           (mean(testNPV), std(testNPV)),
           (mean(testF1), std(testF1)),
           testConfMatrix
end
