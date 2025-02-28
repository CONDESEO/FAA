using Random

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
    
    #Extraer datos de entrenamiento
    inputs = trainingDataset[:,1:4]
    targets = trainingDataset[:,5]
    #convertimos a float para facilitar a Flux
    inputs = Float32.(inputs) 
    targets = Float32.(targets)

    #Extraer datos de validación
    inputsVal = validationDataset[:,1:4]
    targetsVal = validationDataset[:,5]
    inputsVal = Float32.(inputsVal)
    targetsVal = Float32.(targetsVal)

    #Extraer datos de test
    inputsTest = testDataset[:,1:4]
    targetsTest = testDataset[:,5]
    inputsTest = Float32.(inputsTest)
    targetsTest = Float32.(targetsTest)

    #sacamos las neuronas que deben tener la capa de entrada y salida
    numInputs = size(inputs, 2)
    numOutputs = size(targets, 2)

    #llamamos a la funcion que crea la RNA con la topología
    ann = buildClassANN(numInputs, topology, numOutputs; transferFunctions=transferFunctions)
    
    #Inicializamos el historial de pérdidas
    lossHistory = Float32[loss(inputs', targets')]
    valLossHistory = isempty(valInputs) ? Float32[] : Float32[loss(valInputs', valTargets')]
    testLossHistory = isempty(testInputs) ? Float32[] : Float32[loss(testInputs', testTargets')]
    
    #funcion de pérdida que usamos para entrenar
    loss(x,y) = Flux.Losses.mse(ann(x),y)

    # Elegimos el optimizador Adam con la tasa de aprendizaje que nos pasaron
    opt = Flux.Optimise.Adam(learningRate)

    #obtenemos los parámetros de la red
    ps = Flux.params(ann)

    #Variables para la parada temprana
    bestValLoss = isempty(valLossHistory) ? Inf32 : valLossHistory[1] 
    epochsWithoutImprovement = 0

    #bucle que entrena la red maxEpochs iteraciones o hasta que la pérdida sea suficientemente baja  
    for epoch in 1:maxEpochs
        Flux.train!(loss,ps,[(inputs',targets')], opt) #entrenamos una vez por iteración

        #pérdidas de entrenamiento, validación y test
        trainLoss = loss(inputs', targets')
        push!(lossHistory, trainLoss)
        valLoss = isempty(valInputs) ? Inf32 : loss(valInputs', valTargets')
        push!(valLossHistory, valLoss)
        testLoss = isempty(testInputs) ? Inf32 : loss(testInputs', testTargets')
        push!(testLossHistory, testLoss)

        #Verifivar si hay mejor modelo basado en la validación
        if !isempty(valInputs) 
            if valLoss < bestValLoss
                bestValLoss = valLoss
                epochsWithoutImprovement = 0
            else
                epochsWithoutImprovement += 1
            end

            #Parada temprena si no mejora en maxEpochsVal iteraciones
            if epochsWithoutImprovement >= maxEpochsVal
                break
            end
        end

        #Parada si la péridda de entrenamiento alcanza minLoss
        if trainLoss < minLoss
            break
        end
    end

    #Devolver la mejor RNA y los historiales de pérdidas
    finalAnn = isempty(valInputs) ? ann : bestAnn
    return finalAnn, lossHistory, valLossHistory, testLossHistory
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)

    
end;