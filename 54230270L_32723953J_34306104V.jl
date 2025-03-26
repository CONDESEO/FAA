using Random
using Statistics
using Flux
using Flux.Losses
using SymDoME



#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    numClasses = length(classes)
    numSamples = length(feature)
    
    if (numClasses <= 2)
        # Si solo hay dos clases, se genera una matriz con una columna
        oneHotMatrix = reshape(feature .== classes[1], :, 1)
    else
        # Si hay más de dos clases se genera una matriz con una columna por clase
        oneHotMatrix = convert(BitArray{2}, hcat([instance .== classes for instance in feature]...)')
    end
    
    return oneHotMatrix
end;

function oneHotEncoding(feature::AbstractArray{<:Any,1})
    classes = unique(feature)
    return oneHotEncoding(feature, classes)
end;

function oneHotEncoding(feature::AbstractArray{Bool,1})
    numSamples = length(feature)
    oneHotMatrix = zeros(Int, numSamples, 2)
    
    for i in 1:numSamples
        if feature[i]
            oneHotMatrix[i, 2] = true  # True -> [0, 1]
        else
            oneHotMatrix[i, 1] = true  # False -> [1, 0]
        end
    end
    
    return oneHotMatrix
end;


#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////


function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    MinMaxParameters = (minimum(dataset, dims=1), maximum(dataset, dims=1))
    return MinMaxParameters
end;

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    ZeroMeanParameters = (mean(dataset, dims=1), std(dataset, dims=1))
    return ZeroMeanParameters
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues, maxValues = normalizationParameters
    dataset .= (dataset.- minValues) ./ (maxValues .- minValues)
    dataset[:, vec(minValues .== maxValues)] .= 0 #Si todos los valores son iguales los ponemos a 0
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    normalizeMinMax!(dataset, normalizationParameters)
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) #Usar copy
    minValues, maxValues = normalizationParameters
    normalizedDataset = (dataset .-minValues) ./ (maxValues .- minValues)
    normalizedDataset[:, vec(minValues .== maxValues)] .= 0
    return normalizedDataset
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2}) #Usar copy
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    return normalizeMinMax(dataset, normalizationParameters)
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    meanValues, stdValues = normalizationParameters
    dataset .= (dataset.- meanValues) ./ stdValues
    dataset[:, vec(stdValues .== 0)] .= 0 #Si todos los valores son iguales los ponemos a 0
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset)
    normalizeZeroMean!(dataset, normalizationParameters)
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    meanValues, stdValues = normalizationParameters
    normalizedDataset = (dataset .- meanValues) ./ stdValues
    normalizedDataset[:, vec(stdValues .== 0)] .= 0
    return normalizedDataset
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    normalizedParameters = calculateZeroMeanNormalizationParameters(dataset)
    return normalizeZeroMean(dataset, normalizedParameters)
end;


#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    return  outputs .>= threshold
end;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    numCols = size(outputs, 2)
    if numCols == 1
        return reshape(classifyOutputs(outputs[:]; threshold=threshold), :, 1)
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2)
        classified_outputs = falses(size(outputs))
        classified_outputs[indicesMaxEachInstance] .= true

        return classified_outputs
    end
end



#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; 
                       transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))    ann = Chain()

    numInputsLayer = numInputs

    if !isempty(topology)
        for (i, numOutputsLayer) in enumerate(topology) 
            ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[i]))  
            numInputsLayer = numOutputsLayer  
        end
    end

    if numOutputs > 2
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))  
        ann = Chain(ann..., softmax )
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, σ))  
    end

    return ann
end

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    return mean(outputs .== targets)
end


function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    numCols = size(outputs, 2)

    if numCols == 1
        return accuracy(outputs[:,1], targets[:,1])
    else
        comparisonMatrix = outputs .== targets 
        correctlyClassified = all(comparisonMatrix, dims=2)  
        return mean(correctlyClassified)
    end
end


function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    classified_outputs = classifyOutputs(outputs[:]; threshold=threshold)  
    return accuracy(classified_outputs, targets)
end


function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    numCols = size(outputs, 2)

    if numCols == 1
        return accuracy(outputs[:,1], targets[:,1]; threshold=threshold)
    else
        classified_outputs = classifyOutputs(outputs; threshold=threshold)  
        return accuracy(classified_outputs, targets)  
    end
end


#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////


function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    @assert(numOutputs != 2)
    ann = Chain()

    numInputsLayer = numInputs

    if !isempty(topology)
        for (i, numOutputsLayer) in enumerate(topology) 
            ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[i]))  
            numInputsLayer = numOutputsLayer  
        end
    end

    if numOutputs > 2
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))  
        ann = Chain(ann..., softmax )
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, σ))  
    end

    return ann
end

function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    
    #definimos entradas y salidas del dataset
    inputs = dataset[:,1:4]
    targets = dataset[:,5]

    #convertimos a float para facilitar a Flux
    inputs = Float32.(inputs) 
    targets = Float32.(targets)

    #sacamos las neuronas que deben tener la capa de entrada y salida
    numInputs = size(inputs, 2)
    numOutputs = size(targets, 2)

    #llamamos a la funcion que crea la RNA con la topología
    ann = buildClassANN(numInputs, topology, numOutputs; transferFunctions=transferFunctions)

    lossHistory = Float32[] #guardamos como cambia la pérdida en cada iteración

    #funcion de pérdida que usamos para entrenar
    loss(x,y) = Flux.Losses.mse(ann(x),y)

    # Elegimos el optimizador Adam con la tasa de aprendizaje que nos pasaron
    opt = Flux.Optimise.Adam(learningRate)

    #obtenemos los parámetros de la red
    ps = Flux.params(ann)

    #guardamos el valor de la pérdida antes de empezar el entrenamiento
    push!(lossHistory, loss(inputs', targets'))


    #bucle que entrena la red maxEpochs iteraciones o hasta que la pérdida sea suficientemente baja  
    for epoch in 1:maxEpochs
        Flux.train!(loss,ps,[(inputs',targets')], opt) #entrenamos una vez por iteración

        epochLoss = loss(inputs', targets') #calculamos pérdida después de entrenar
        push!(lossHistory, epochLoss) #guardamos la pérdida

        if epochLoss <= minLoss #si la perdida es menor o igual a minLoss paramos el entrenamiento
            break
        end
    end

    return (ann, lossHistory) #Devolvemos la red entrenada y el historial de pérdida
end

function trainClassANN(topology::AbstractArray{<:Int,1}, (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    targets = reshape(targets, :, 1) #convertimos a una matriz de 1 columna
    return trainClassANN(topology, (inputs, targets); transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate)
end
function holdOut(N::Int, P::Real)
    indices = randperm(N)
    n_test = Int(round(int, P * N))
    indices_test = indices[1:n_test]
    indices_entrenamiento = indices[n_test + 1:end]
    return indices_entrenamiento, indices_test
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    indices = randperm(N)
    n_test = Int(round(int, Ptest * N))
    indices_test = indices[1:n_test]
    train_val_indices = setdiff(indices, indices_test)
    n_validacion = Int(round(int, Pval * length(train_val_indices)))
    indices_validacion = train_val_indices[1: n_validacion]
    indices_entrenamiento = train_val_indices[n_validacion + 1: end]
    return indices_validacion, indices_entrenamiento, indices_test
end;
function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
 
    (trainingInputs,   trainingTargets)   = trainingDataset;
    (validationInputs, validationTargets) = validationDataset;
    (testInputs,       testTargets)       = testDataset;
 
    # Se supone que tenemos cada patron en cada fila
    # Comprobamos que el numero de filas (numero de patrones) coincide tanto en entrenamiento como en validacion como test
    @assert(size(trainingInputs,   1)==size(trainingTargets,   1));
    @assert(size(testInputs,       1)==size(testTargets,       1));
    @assert(size(validationInputs, 1)==size(validationTargets, 1));
    # Comprobamos que el numero de columnas coincide en los grupos de entrenamiento y validación, si este no está vacío
    !isempty(validationInputs)  && @assert(size(trainingInputs, 2)==size(validationInputs, 2));
    !isempty(validationTargets) && @assert(size(trainingTargets,2)==size(validationTargets,2));
    # Comprobamos que el numero de columnas coincide en los grupos de entrenamiento y test, si este no está vacío
    !isempty(testInputs)  && @assert(size(trainingInputs, 2)==size(testInputs, 2));
    !isempty(testTargets) && @assert(size(trainingTargets,2)==size(testTargets,2));
 
    # Pasamos los datos a Float32
    trainingInputs   = Float32.(  trainingInputs);
    validationInputs = Float32.(validationInputs);
    testInputs       = Float32.(      testInputs);
 
    # Creamos la RNA
    ann = buildClassANN(size(trainingInputs,2), topology, size(trainingTargets,2); transferFunctions=transferFunctions);
 
    # Definimos la funcion de loss
    loss(model,x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);
 
    # Creamos los vectores con los valores de loss y de precision en cada ciclo
    trainingLosses   = Float32[];
    validationLosses = Float32[];
    testLosses       = Float32[];
 
    # Empezamos en el ciclo 0
    numEpoch = 0;
 
    # Una funcion util para calcular los resultados y mostrarlos por pantalla si procede
    function calculateLossValues()
        # Calculamos el loss en entrenamiento, validacion y test. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        trainingLoss = loss(ann, trainingInputs', trainingTargets');
        validationLoss = NaN; testLoss = NaN;
        push!(trainingLosses, trainingLoss);
        !isempty(validationInputs) && (validationLoss = loss(ann, validationInputs', validationTargets'); push!(validationLosses, validationLoss);)
        !isempty(      testInputs) && (testLoss       = loss(ann,       testInputs',       testTargets'); push!(      testLosses,       testLoss);)
        return (trainingLoss, validationLoss, testLoss);
    end;
 
    # Calculamos los valores de loss para el ciclo 0 (sin entrenar nada)
    (trainingLoss, validationLoss, _) = calculateLossValues();
 
    if isempty(validationInputs) maxEpochsVal=Inf; end;
 
    # Numero de ciclos sin mejorar el error de validacion y el mejor error de validation encontrado hasta el momento
    numEpochsValidation = 0; bestValidationLoss = validationLoss;
    # Cual es la mejor ann que se ha conseguido
    bestANN = deepcopy(ann);
 
    opt_state = Flux.setup(Adam(learningRate), ann)
 
    # Entrenamos hasta que se cumpla una condicion de parada
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss) && (numEpochsValidation<maxEpochsVal)
 
        # Entrenamos 1 ciclo. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        Flux.train!(loss, ann, [(trainingInputs', trainingTargets')], opt_state);
 
        # Aumentamos el numero de ciclo en 1
        numEpoch += 1;
 
        # Calculamos los valores de loss para este ciclo
        (trainingLoss, validationLoss, _) = calculateLossValues();
 
        # Aplicamos la parada temprana si hay conjunto de validacion
        if !isempty(validationInputs)
            if validationLoss<bestValidationLoss
                bestValidationLoss = validationLoss;
                numEpochsValidation = 0;
                bestANN = deepcopy(ann);
            else
                numEpochsValidation += 1;
            end;
        end;
 
    end;
 
    # Si no hubo conjunto de validacion, la mejor RNA será siempre la del último ciclo
    if isempty(validationInputs)
        bestANN = ann;
    end;
 
    return (bestANN, trainingLosses, validationLosses, testLosses);
end;


function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
 
    (trainingInputs,   trainingTargets)   = trainingDataset;
    (validationInputs, validationTargets) = validationDataset;
    (testInputs,       testTargets)       = testDataset;
 
    return trainClassANN(topology, (trainingInputs, reshape(trainingTargets, length(trainingTargets), 1)); validationDataset=(validationInputs, reshape(validationTargets, length(validationTargets), 1)), testDataset=(testInputs, reshape(testTargets, length(testTargets), 1)), transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal);
end;

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    numInstances = length(targets);
    @assert(length(outputs)==numInstances);
    # Valores de la matriz de confusion
    TN = sum(.!outputs .& .!targets); # Verdaderos negativos
    FN = sum(.!outputs .&   targets); # Falsos negativos
    TP = sum(  outputs .&   targets); # Verdaderos positivos
    FP = sum(  outputs .& .!targets); # Falsos negativos
    # Creamos la matriz de confusión, poniendo en las filas los que pertenecen a cada clase (targets) y en las columnas los clasificados (outputs)
    #  Primera fila/columna: negativos
    #  Segunda fila/columna: positivos
    confMatrix = [TN FP;
                  FN TP];
    # Metricas que se derivan de la matriz de confusion:
    acc         = (TN+TP)/(TN+FN+TP+FP);
    errorRate   = 1. - acc;
    # Para sensibilidad, especificidad, VPP y VPN controlamos que algunos casos pueden ser NaN
    #  Para el caso de sensibilidad y especificidad, en un conjunto de entrenamiento estos no pueden ser NaN, porque esto indicaria que se ha intentado entrenar con una unica clase
    #   Sin embargo, sí pueden ser NaN en el caso de aplicar un modelo en un conjunto de test, si este sólo tiene patrones de una clase
    #  Para VPP y VPN, sí pueden ser NaN en caso de que el clasificador lo haya clasificado todo como negativo o positivo respectivamente
    # En estos casos, estas metricas habria que dejarlas a NaN para indicar que no se han podido evaluar
    #  Sin embargo, como es posible que se quiera combinar estos valores al evaluar una clasificacion multiclase, es necesario asignarles un valor. El criterio que se usa aqui es que estos valores seran igual a 1
    #   Se utiliza este criterio porque, por ejemplo en el caso de recall (sensibilidad), no habría fallado en ningún positivo, porque no hay ninguno
    recall      = (TP==FN==0) ? 1. : TP/(TP+FN); # Sensibilidad
    specificity = (TN==FP==0) ? 1. : TN/(TN+FP); # Especificidad
    precision   = (TP==FP==0) ? 1. : TP/(TP+FP); # Valor predictivo positivo
    NPV         = (TN==FN==0) ? 1. : TN/(TN+FN); # Valor predictivo negativo
    # Calculamos F1
    F1          = (recall==precision==0) ? 0. : 2*(recall*precision)/(recall+precision);
    @assert(!isnan(F1));
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end;
 
 
confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) = confusionMatrix(outputs.>=threshold, targets);
 
 
function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert(size(outputs)==size(targets));
    (numInstances, numClasses) = size(targets);
    # Nos aseguramos de que no hay dos columnas
    @assert(numClasses!=2);
    if (numClasses==1)
        return confusionMatrix(outputs[:,1], targets[:,1]);
    end;
 
    # Nos aseguramos de que en cada fila haya uno y sólo un valor a true
    @assert(all(sum(outputs, dims=2).==1));
    # Reservamos memoria para las metricas de cada clase, inicializandolas a 0 porque algunas posiblemente no se calculen
    recall      = zeros(numClasses);
    specificity = zeros(numClasses);
    precision   = zeros(numClasses);
    NPV         = zeros(numClasses);
    F1          = zeros(numClasses);
    # Calculamos las metricas para cada clase, usando la función anterior para problemas de 2 clases
    for numClass in 1:numClasses
        # Calculamos las metricas de cada problema binario correspondiente a cada clase y las almacenamos en los vectores correspondientes
        (_, _, recall[numClass], specificity[numClass], precision[numClass], NPV[numClass], F1[numClass], _) = confusionMatrix(outputs[:,numClass], targets[:,numClass]);
    end;
    # Es posible hacer esta operación sin bucle, de esta manera:
    # [(_, _, recall[numClass], specificity[numClass], precision[numClass], NPV[numClass], F1[numClass], _) = confusionMatrix(outputs[:,numClass], targets[:,numClass]); for numClass in 1:numClasses]
 
    # Creamos la matriz de confusión
    confMatrix = [sum(targets[:,numClassTarget] .& outputs[:,numClassOutput]) for numClassTarget in 1:numClasses, numClassOutput in 1:numClasses];
 
    # Aplicamos las formas de combinar las metricas macro o weighted
    if weighted
        # Calculamos los valores de ponderacion para hacer el promedio
        numInstancesFromEachClass = vec(sum(targets, dims=1));
        @assert(numInstances == sum(numInstancesFromEachClass));
        weights = numInstancesFromEachClass./sum(numInstancesFromEachClass);
        recall      = sum(weights.*recall);
        specificity = sum(weights.*specificity);
        precision   = sum(weights.*precision);
        NPV         = sum(weights.*NPV);
        F1          = sum(weights.*F1);
    else
        recall      = mean(recall);
        specificity = mean(specificity);
        precision   = mean(precision);
        NPV         = mean(NPV);
        F1          = mean(F1);
    end;
    # Precision y tasa de error las calculamos con las funciones definidas previamente
    acc = accuracy(outputs, targets);
    errorRate = 1 - acc;
 
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
end;
 
confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true) = confusionMatrix(classifyOutputs(outputs; threshold=threshold), targets; weighted=weighted)
 
 
 
 
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Comprobamos que todas las clases de salida esten dentro de las clases de las salidas deseadas
    @assert(all([in(label, classes) for label in vcat(targets, outputs)]));
    # Es importante pasar el mismo vector de clases como argumento a las 2 llamadas a oneHotEncoding para que el orden de las clases sea el mismo en ambas matrices
    return confusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);
end;
 
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs));
    return confusionMatrix(outputs, targets, classes; weighted=weighted);
end;
 
 
 
# Funciones auxiliares para visualizar por pantalla la matriz de confusion y las metricas que se derivan de ella
function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets; weighted=weighted);
    numClasses = size(confMatrix,1);
    writeHorizontalLine() = (for i in 1:numClasses+1 print("--------") end; println(""); );
    writeHorizontalLine();
    print("\t| ");
    if (numClasses==2)
        println(" - \t + \t|");
    else
        print.("Cl. ", 1:numClasses, "\t| ");
    end;
    println("");
    writeHorizontalLine();
    for numClassTarget in 1:numClasses
        # print.(confMatrix[numClassTarget,:], "\t");
        if (numClasses==2)
            print(numClassTarget == 1 ? " - \t| " : " + \t| ");
        else
            print("Cl. ", numClassTarget, "\t| ");
        end;
        print.(confMatrix[numClassTarget,:], "\t| ");
        println("");
        writeHorizontalLine();
    end;
    println("Accuracy: ", acc);
    println("Error rate: ", errorRate);
    println("Recall: ", recall);
    println("Specificity: ", specificity);
    println("Precision: ", precision);
    println("Negative predictive value: ", NPV);
    println("F1-score: ", F1);
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
end;
printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) =  printConfusionMatrix(classifyOutputs(outputs), targets; weighted=weighted)
 
 
 
printConfusionMatrix(outputs::AbstractArray{Bool,1},   targets::AbstractArray{Bool,1})                      = printConfusionMatrix(reshape(outputs, :, 1), reshape(targets, :, 1));
printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) = printConfusionMatrix(outputs.>=threshold,    targets);
 
printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true) = printConfusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);
 
function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs));
    printConfusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);
end;    



function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainInputs = Float64.(trainingDataset[1])
    trainTargets = trainingDataset[2]
    testInputs = Float64.(testInputs)
    _, _, _, model = dome(trainInputs, trainTargets;
    maximumNodes = maximumNodes)
    model_str = string(model)
    testOutputs = evaluateTree(model, testInputs)
    
    if isa(testOutputs, Real)
        testOutputs = repeat([testOutputs], size(testInputs,1))
    end

    return testOutputs
end

        
function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)

    if size(trainingDataset[2], 2) == 1
        trainingDataset = (trainingDataset[1], reshape(trainingDataset[2], :, 1))
        return trainClassDoME(trainingDataset, testInputs, maximumNodes)
    end

    num_classes = size(trainingDataset[2], 2)
    num_rows = size(testInputs, 1)
    matrix = Matrix{Float64}(undef, num_rows, num_classes)
    for i in 1:num_classes
        new_targets = trainingDataset[2][:, i]
        matrix[:, i] = trainClassDoME((trainingDataset[1], new_targets), testInputs, maximumNodes)
    end

    return matrix
end

        
function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    classes = unique(trainingDataset[2])
    num_classes = length(classes)
    testOutputs = Array{eltype(trainingDataset[2]),1}(undef, size(testInputs,1))
    testOutputsDoME = trainClassDoME( (trainingDataset[1], oneHotEncoding(trainingDataset[2], classes)), 
    testInputs, maximumNodes)
    testOutputsBool = classifyOutputs(testOutputsDoME; threshold=0)

    if num_classes == 2
        testOutputsBool = vec(testOutputsBool)  # Asegura que sea un vector
        testOutputs[testOutputsBool] .= classes[1]
        testOutputs[.!testOutputsBool] .= classes[2]
    else
        for numClass in 1:num_classes
            testOutputs[testOutputsBool[:, numClass]] .= classes[numClass]
        end
    end

    return testOutputs
end

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////


function crossvalidation(N::Int64, k::Int64)
    @assert(k >= 10)
    vector = collect(1:k)
    num_reps = ceil(Int, N/k)
    v_repeated = repeat(vector, num_reps)
    v_repeated = v_repeated[1:N]
    shuffle!(v_repeated)
    return v_repeated
end

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    @assert(k >= 10)
    index_vector = collect(1:size(targets))
    num_positive_instances = sum(targets)
    index_vector[targets] .= crossvalidation(num_positive_instances, k)
    index_vector[.!targets] .= crossvalidation(size(targets) - num_positive_instances, k)
    return index_vector
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    @assert(k >= 10)
    index_vector = collect(1:size(targets, 1))
    num_classes = size(targets, 2)

    for i in 1:num_classes
        index_vector[targets[:, i]] .= crossvalidation(sum(targets[:, i]), k)
    end

    return index_vector

end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    one_hot_targets = oneHotEncoding(targets)
    return crossvalidation(one_hot_targets, k)
end


function ANNCrossValidation(
    topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    validationRatio::Real=0, maxEpochsVal::Int=20
)   
    # Obtener las clases y codificación one-hot para las etiquetas
    inputs  = dataset[1]
    targets = dataset[2]  
    classes = unique(targets)
    num_classes = length(classes)
    one_hot_targets = oneHotEncoding(targets, classes)
    
    num_folds = maximum(crossValidationIndices)  # El número de folds

    # Inicializar vectores para métricas
    accuracies = zeros(num_folds)
    error_rates = zeros(num_folds)
    recalls = zeros(num_folds)
    specificities = zeros(num_folds)
    precisions = zeros(num_folds)
    npvs = zeros(num_folds)
    f1_scores = zeros(num_folds)
    cm = zeros(Float64, num_classes, num_classes, num_folds)  # Matriz de confusión

    # k-fold cross-validation
    for i in 1:num_folds
        # Separar índices de test y entrenamiento
        test_indices = findall(x -> x == i, crossValidationIndices)
        train_indices = findall(x -> x != i, crossValidationIndices)
    
        if validationRatio > 0
            real_ratio = validationRatio / (1 - 1/num_folds)
            train_indices_relative, val_indices_relative = holdOut(length(train_indices), real_ratio)
            val_indices = train_indices[val_indices_relative]
            train_indices = train_indices[train_indices_relative]
            validationInputs = inputs[val_indices, :]
            validationTargets = one_hot_targets[val_indices, :]
        else
            val_indices = Int[]
            validationInputs = Array{Float64, 2}(undef, 0, size(inputs, 2))
            validationTargets = Array{Bool, 2}(undef, 0, size(one_hot_targets, 2))
        end

        # Extraer datos de entrenamiento y prueba correctamente
        trainingInputs = inputs[train_indices, :]
        trainingTargets = one_hot_targets[train_indices, :]
        testInputs = inputs[test_indices, :]
        testTargets = one_hot_targets[test_indices, :]

        # Ejecutar el entrenamiento numExecutions veces
        for _ in 1:numExecutions
            # Entrenar el modelo incluyendo validación y test
            model, _ = trainClassANN(
                topology, 
                (trainingInputs, trainingTargets);
                validationDataset = (validationInputs, validationTargets),
                testDataset = (testInputs, testTargets),
                transferFunctions = transferFunctions,
                maxEpochs = maxEpochs,
                learningRate = learningRate
            )

            # Evaluar en test
            predictions = model(testInputs')'
            acc, err, recall, specificity, precision, npv, f1_score, cm_iter = confusionMatrix(predictions, testTargets)

            # Almacenar métricas
            accuracies[i] += acc / numExecutions
            error_rates[i] += err / numExecutions
            recalls[i] += recall / numExecutions
            specificities[i] += specificity / numExecutions
            precisions[i] += precision / numExecutions
            npvs[i] += npv / numExecutions
            f1_scores[i] += f1_score / numExecutions
            cm[:, :, i] += cm_iter 
        end
    end

    # Calcular la media y desviación estándar de las métricas
    function compute_stats(values)
        return (mean(values), std(values))
    end

    accuracy_stats = compute_stats(accuracies)
    error_rate_stats = compute_stats(error_rates)
    recall_stats = compute_stats(recalls)
    specificity_stats = compute_stats(specificities)
    npv_stats = compute_stats(npvs)
    precision_stats = compute_stats(precisions)
    f1_score_stats = compute_stats(f1_scores)

    # Promediar la matriz de confusión global
    cm_mean = mean(cm, dims=3)

    println("Accuracy: $(accuracy_stats[1]) ± $(accuracy_stats[2])")
    println("Error rate: $(error_rate_stats[1]) ± $(error_rate_stats[2])")
    println("Recall: $(recall_stats[1]) ± $(recall_stats[2])")
    println("Specificity: $(specificity_stats[1]) ± $(specificity_stats[2])")
    println("Precision: $(precision_stats[1]) ± $(precision_stats[2])")
    println("NPV: $(npv_stats[1]) ± $(npv_stats[2])")
    println("F1 Score: $(f1_score_stats[1]) ± $(f1_score_stats[2])")
    println("Confusion Matrix:")
    println(cm_mean)

    return (
        accuracy_stats, error_rate_stats, recall_stats,
        specificity_stats, precision_stats, npv_stats,
        f1_score_stats, cm_mean
    )
end