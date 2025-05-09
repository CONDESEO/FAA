using DataFrames
using SymDoME

function oneHotEncoding(col::AbstractVector)
    clases = unique(col)
    df = DataFrame()
    for clase in clases
        df[!, Symbol(string(clase))] = col .== clase
    end
    return df
end

function oneHotEncoding(col::Vector, clases::Vector, colName::String)
    df_encoded = DataFrame()
    for clase in clases
        # Construimos el nombre de la columna: "nombrecol_codificada_<clase>"
        nuevo_nombre = Symbol("$(colName)_codificada_$(clase)")
        df_encoded[!, nuevo_nombre] = col .== clase
    end
    return df_encoded
end


function normalizeZeroMean(v::Vector{<:Real})
    media = mean(v)
    desviacion = std(v)
    return (v .- media) ./ desviacion
end

function crossvalidation(N::Int64, k::Int64)
    indices = repeat(1:k, Int64(ceil(N/k)));
    indices = indices[1:N];
    shuffle!(indices);
    return indices;
end;

# Validación cruzada estratificada para clasificación binaria o multiclase
function crossvalidation(targets::AbstractArray{Bool,1}, k::Int)
    indices = Array{Int64,1}(undef, length(targets))
    indices[targets] = crossvalidation(sum(targets), k)
    indices[.!targets] = crossvalidation(sum(.!targets), k)
    return indices
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int)
    numClasses = size(targets, 2)
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
crossvalidation(targets::AbstractArray{<:Any,1}, k::Int) = crossvalidation(Matrix(oneHotEncoding(targets)), k)




#////////////////////////////////////////////////////////////////////////////


# -------------------------------------------------------
# Funciones para crear y entrenar una RNA

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    ann=Chain();
    numInputsLayer = numInputs;
    for numHiddenLayer in 1:length(topology)
        numNeurons = topology[numHiddenLayer];
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]));
        numInputsLayer = numNeurons;
    end;
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end;
    return ann;
end;

σ(x) = 1 / (1 + exp(-x))

function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    (inputs, targets) = dataset;

    # Se supone que tenemos cada patron en cada fila
    # Comprobamos que el numero de filas (numero de patrones) coincide
    @assert(size(inputs,1)==size(targets,1));

    # Pasamos los datos a Float32
    inputs = Float32.(inputs);

    # Creamos la RNA
    ann = buildClassANN(size(inputs,2), topology, size(targets,2));

    # Definimos la funcion de loss
    loss(model,x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);

    # Creamos los vectores con los valores de loss y de precision en cada ciclo
    trainingLosses = Float32[];

    # Empezamos en el ciclo 0
    numEpoch = 0;
    # Calculamos el loss para el ciclo 0 (sin entrenar nada)
    trainingLoss = loss(ann, inputs', targets');
    #  almacenamos el valor de loss y precision en este ciclo
    push!(trainingLosses, trainingLoss);
    #  y lo mostramos por pantalla
    println("Epoch ", numEpoch, ": loss: ", trainingLoss);

    opt_state = Flux.setup(Adam(learningRate), ann);

    # Entrenamos hasta que se cumpla una condicion de parada
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss)

        # Entrenamos 1 ciclo. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        Flux.train!(loss, ann, [(inputs', targets')], opt_state);

        # Aumentamos el numero de ciclo en 1
        numEpoch += 1;
        # Calculamos las metricas en este ciclo
        trainingLoss = loss(ann, inputs', targets');
        #  almacenamos el valor de loss
        push!(trainingLosses, trainingLoss);
        #  lo mostramos por pantalla
        println("Epoch ", numEpoch, ": loss: ", trainingLoss);

    end;

    # Devolvemos la RNA entrenada y el vector con los valores de loss
    return (ann, trainingLosses);
end;


trainClassANN(topology::AbstractArray{<:Int,1}, (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01) = trainClassANN(topology, (inputs, reshape(targets, length(targets), 1)); maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate)





#///////////////////////////////////////////////////////////////////////////////////////




function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)

    (trainingInputs, trainingTargets) = trainingDataset;

    trainingInputs = Float64.(trainingInputs);
    testInputs = Float64.(testInputs);

    _, _, _, model = dome(trainingInputs, trainingTargets;
        maximumNodes = maximumNodes
    )

    # El resultado de evaluateTree puede ser un valor real, o un vector
    testOutputs = evaluateTree(model, testInputs);
    if isa(testOutputs, Real)
        testOutputs = repeat([testOutputs], size(testInputs,1));
    end;
    return testOutputs;
end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)

    (trainingInputs, trainingTargets) = trainingDataset;
    numColumns = size(trainingTargets,2);

    if numColumns==1

        testOutputs = trainClassDoME((trainingInputs, vec(trainingTargets)), testInputs, maximumNodes);
        return reshape(testOutputs, :, 1)

    end;

    # Nos aseguramos de que hay mas de dos clases
    @assert(numColumns>2);

    testOutputs = Array{Float64,2}(undef, size(testInputs,1), numColumns);
    for numClass in Base.OneTo(numColumns)

        testOutputs[:,numClass] .= trainClassDoME((trainingInputs, trainingTargets[:,numClass]), testInputs, maximumNodes)

    end;
    return testOutputs
end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)

    (trainingInputs, trainingTargets) = trainingDataset;
    classes = unique(trainingTargets);

    testOutputsDoME = trainClassDoME((trainingInputs, oneHotEncoding(trainingTargets, classes)), testInputs, maximumNodes);
    testOutputsBool = classifyOutputs(testOutputsDoME; threshold=0);

    testOutputs = Array{eltype(trainingTargets),1}(undef, size(testInputs,1));

    if length(classes)<=2
        @assert(isa(testOutputsBool, Vector) || size(testOutputsBool,2)==1)
        testOutputsBool = vec(testOutputsBool);
        testOutputs[  testOutputsBool] .= classes[1];
        if length(classes)==2
            testOutputs[.!testOutputsBool] .= classes[2];
        else @assert(all(testOutputsBool))
        end;
    else
        @assert(all(sum(testOutputsBool, dims=2).==1));
        for numClass in eachindex(classes)
            testOutputs[testOutputsBool[:,numClass]] .= classes[numClass];
        end;
    end;
    return testOutputs;
end;


#/////////////////////////////////////////////////////////////////////////////////////////



function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20)

    (inputs,targets) = dataset;

    # Comprobamos que el numero de patrones coincide
    @assert(size(inputs,1)==length(targets));

    # Que clases de salida tenemos
    classes = unique(targets);

    # Primero codificamos las salidas deseadas en caso de entrenar RR.NN.AA.
    targets = oneHotEncoding(targets, classes);

    # Creamos los vectores para las metricas que se vayan a usar
    numFolds = maximum(crossValidationIndices);
    testAccuracy    = Array{Float64,1}(undef, numFolds);
    testErrorRate   = Array{Float64,1}(undef, numFolds);
    testRecall      = Array{Float64,1}(undef, numFolds);
    testSpecificity = Array{Float64,1}(undef, numFolds);
    testPrecision   = Array{Float64,1}(undef, numFolds);
    testNPV         = Array{Float64,1}(undef, numFolds);
    testF1          = Array{Float64,1}(undef, numFolds);
    testConfusionMatrix = zeros(length(classes), length(classes));

    # Para cada fold, entrenamos
    for numFold in 1:numFolds

        # Dividimos los datos en entrenamiento y test
        trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
        testInputs        = inputs[crossValidationIndices.==numFold,:];
        trainingTargets   = targets[crossValidationIndices.!=numFold,:];
        testTargets       = targets[crossValidationIndices.==numFold,:];

        # Como el entrenamiento de RR.NN.AA. es no determinístico, hay que entrenar varias veces, y
        #  se crean vectores adicionales para almacenar las metricas para cada entrenamiento
        testAccuracyEachRepetition        = Array{Float64,1}(undef, numExecutions);
        testErrorRateEachRepetition       = Array{Float64,1}(undef, numExecutions);
        testRecallEachRepetition          = Array{Float64,1}(undef, numExecutions);
        testSpecificityEachRepetition     = Array{Float64,1}(undef, numExecutions);
        testPrecisionEachRepetition       = Array{Float64,1}(undef, numExecutions);
        testNPVEachRepetition             = Array{Float64,1}(undef, numExecutions);
        testF1EachRepetition              = Array{Float64,1}(undef, numExecutions);
        testConfusionMatrixEachRepetition = Array{Float64,3}(undef, length(classes), length(classes), numExecutions);

        # Se entrena las veces que se haya indicado
        for numTraining in 1:numExecutions

            if validationRatio>0

                # Para el caso de entrenar una RNA con conjunto de validacion, hacemos una división adicional:
                #  dividimos el conjunto de entrenamiento en entrenamiento+validacion
                #  Para ello, hacemos un hold out
                (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), validationRatio*size(inputs,1)/size(trainingInputs,1));
                # Con estos indices, se pueden crear los vectores finales que vamos a usar para entrenar una RNA
                # Otra forma de hacer el mismo cálculo:
                # (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), validationRatio*numFolds/(numFolds-1));

                # Entrenamos la RNA, teniendo cuidado de codificar las salidas deseadas correctamente
                ann, = trainClassANN(topology, (trainingInputs[trainingIndices,:],   trainingTargets[trainingIndices,:]),
                    validationDataset = (trainingInputs[validationIndices,:], trainingTargets[validationIndices,:]),
                    testDataset =       (testInputs,                          testTargets);
                    transferFunctions = transferFunctions,
                    maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal);
                    
            else

                # Si no se desea usar conjunto de validacion, se entrena unicamente con conjuntos de entrenamiento y test,
                #  teniendo cuidado de codificar las salidas deseadas correctamente
                ann, = trainClassANN(topology, (trainingInputs, trainingTargets),
                    testDataset = (testInputs,     testTargets);
                    transferFunctions=transferFunctions,
                    maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate);

            end;

            # Calculamos las metricas correspondientes con la funcion desarrollada en el ejercicio anterior
            (testAccuracyEachRepetition[numTraining], testErrorRateEachRepetition[numTraining], testRecallEachRepetition[numTraining], testSpecificityEachRepetition[numTraining], testPrecisionEachRepetition[numTraining], testNPVEachRepetition[numTraining], testF1EachRepetition[numTraining], testConfusionMatrixEachRepetition[:,:,numTraining]) =
                confusionMatrix(collect(ann(Float32.(testInputs'))'), testTargets);

            @assert( isapprox( testAccuracyEachRepetition[numTraining], sum([testConfusionMatrixEachRepetition[numClass,numClass,numTraining] for numClass in 1:length(classes)])/sum(testConfusionMatrixEachRepetition[:,:,numTraining]) ) );

        end;

        # Almacenamos las metricas como una media de las obtenidas en los entrenamientos de este fold
        testAccuracy[numFold]    = mean(testAccuracyEachRepetition);
        testErrorRate[numFold]   = mean(testErrorRateEachRepetition);
        testRecall[numFold]      = mean(testRecallEachRepetition);
        testSpecificity[numFold] = mean(testSpecificityEachRepetition);
        testPrecision[numFold]   = mean(testPrecisionEachRepetition);
        testNPV[numFold]         = mean(testNPVEachRepetition);
        testF1[numFold]          = mean(testF1EachRepetition);
        testConfusionMatrix    .+= mean(testConfusionMatrixEachRepetition, dims=3)[:,:,1];

    end; # for numFold in 1:numFolds

    return (mean(testAccuracy), std(testAccuracy)), (mean(testErrorRate), std(testErrorRate)), (mean(testRecall), std(testRecall)), (mean(testSpecificity), std(testSpecificity)), (mean(testPrecision), std(testPrecision)), (mean(testNPV), std(testNPV)), (mean(testF1), std(testF1)), testConfusionMatrix;

end;



#////////////////////////////////////////////////////////////////////////////////////////////





SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
DTClassifier  = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0

function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, crossValidationIndices::Array{Int64,1})

    (inputs, targets) = dataset;

    # Comprobamos que el numero de patrones coincide
    @assert(size(inputs,1)==length(targets));

    # Vamos a usar RR.NN.AA.
    if (modelType==:ANN)

        return ANNCrossValidation(modelHyperparameters["topology"],
            dataset, crossValidationIndices;
            numExecutions     = haskey(modelHyperparameters, "numExecutions")     ? modelHyperparameters["numExecutions"]     : 50,
            transferFunctions = haskey(modelHyperparameters, "transferFunctions") ? modelHyperparameters["transferFunctions"] : fill(σ, length(modelHyperparameters["topology"])),
            maxEpochs         = haskey(modelHyperparameters, "maxEpochs")         ? modelHyperparameters["maxEpochs"]         : 1000,
            minLoss           = haskey(modelHyperparameters, "minLoss")           ? modelHyperparameters["minLoss"]           : 0.0,
            learningRate      = haskey(modelHyperparameters, "learningRate")      ? modelHyperparameters["learningRate"]      : 0.01,
            validationRatio   = haskey(modelHyperparameters, "validationRatio")   ? modelHyperparameters["validationRatio"]   : 0,
            maxEpochsVal      = haskey(modelHyperparameters, "maxEpochsVal")      ? modelHyperparameters["maxEpochsVal"]      : 20);

    end;

    # No estamos trabajando con redes de neuronas

    targets = string.(targets);

    # Que clases de salida tenemos
    classes = unique(targets);

    # Creamos los vectores para las metricas que se vayan a usar
    numFolds = maximum(crossValidationIndices);
    testAccuracy        = Array{Float64,1}(undef, numFolds);
    testErrorRate       = Array{Float64,1}(undef, numFolds);
    testRecall          = Array{Float64,1}(undef, numFolds);
    testSpecificity     = Array{Float64,1}(undef, numFolds);
    testPrecision       = Array{Float64,1}(undef, numFolds);
    testNPV             = Array{Float64,1}(undef, numFolds);
    testF1              = Array{Float64,1}(undef, numFolds);
    # testConfusionMatrix = Array{Float64,3}(undef, length(classes), length(classes), numFolds);
    testConfusionMatrix = zeros(Int, length(classes), length(classes));

    # Para cada fold, entrenamos
    for numFold in 1:numFolds

        # Dividimos los datos en entrenamiento y test
        trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
        testInputs        = inputs[crossValidationIndices.==numFold,:];
        trainingTargets   = targets[crossValidationIndices.!=numFold];
        testTargets       = targets[crossValidationIndices.==numFold];

        # Creamos el modelo según lo que nos hayan pasado como parámetro
        if modelType==:DoME

            testOutputs = trainClassDoME((trainingInputs, trainingTargets), testInputs, modelHyperparameters["maximumNodes"]);

        else

            if modelType==:SVC
                @assert((modelHyperparameters["kernel"] == "linear") || (modelHyperparameters["kernel"] == "poly") || (modelHyperparameters["kernel"] == "rbf") || (modelHyperparameters["kernel"] == "sigmoid"));
                model = SVMClassifier(
                    kernel = 
                        modelHyperparameters["kernel"]=="linear"  ? LIBSVM.Kernel.Linear :
                        modelHyperparameters["kernel"]=="rbf"     ? LIBSVM.Kernel.RadialBasis :
                        modelHyperparameters["kernel"]=="poly"    ? LIBSVM.Kernel.Polynomial :
                        modelHyperparameters["kernel"]=="sigmoid" ? LIBSVM.Kernel.Sigmoid : nothing,
                    cost   = Float64(modelHyperparameters["C"]),
                    gamma  = Float64(get(modelHyperparameters, "gamma",  -1)),
                    degree = Int32(  get(modelHyperparameters, "degree", -1)),
                    coef0  = Float64(get(modelHyperparameters, "coef0",  -1)));
                # Cuidado con los tipos de los argumentos cost, gamma, degree y coef0, tienen que ser esos. No vale, por ejemplo, que degree sea Int, tiene que ser Int32

            elseif modelType==:DecisionTreeClassifier
                model = DTClassifier(max_depth = modelHyperparameters["max_depth"], rng=Random.MersenneTwister(1));
            elseif modelType==:KNeighborsClassifier
                model = kNNClassifier(K = modelHyperparameters["n_neighbors"]);
            else
                error(string("Unknown model ", modelType));
            end;

            # Creamos el objeto de tipo Machine
            mach = machine(model, MLJ.table(trainingInputs), categorical(trainingTargets));

            # Entrenamos el modelo con el conjunto de entrenamiento
            MLJ.fit!(mach, verbosity=0)

            # Pasamos el conjunto de test
            testOutputs = MLJ.predict(mach, MLJ.table(testInputs))
            # if modelType==:DecisionTreeClassifier || modelType==:KNeighborsClassifier
            if modelType!=:SVC
                testOutputs = mode.(testOutputs)
            end;
            # testOutputs = string.(testOutputs);

        end;

        # Calculamos las metricas y las almacenamos en las posiciones de este fold de cada vector
        (testAccuracy[numFold], testErrorRate[numFold], testRecall[numFold], testSpecificity[numFold], testPrecision[numFold], testNPV[numFold], testF1[numFold], testConfusionMatrixThisFold) =
            confusionMatrix(testOutputs, testTargets, classes);

        @assert( isapprox( testAccuracy[numFold], sum([testConfusionMatrixThisFold[numClass,numClass] for numClass in 1:length(classes)])/sum(testConfusionMatrixThisFold) ) );

        testConfusionMatrix .+= testConfusionMatrixThisFold;

    end; # for numFold in 1:numFolds

    return (mean(testAccuracy), std(testAccuracy)), (mean(testErrorRate), std(testErrorRate)), (mean(testRecall), std(testRecall)), (mean(testSpecificity), std(testSpecificity)), (mean(testPrecision), std(testPrecision)), (mean(testNPV), std(testNPV)), (mean(testF1), std(testF1)), testConfusionMatrix;

end;
