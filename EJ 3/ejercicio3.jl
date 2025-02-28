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
    
    # Extraer datos correctamente
    inputs, targets = trainingDataset
    inputsVal, targetsVal = validationDataset
    inputsTest, targetsTest = testDataset

    # Convertir datos a Float32 para compatibilidad con Flux
    inputs, targets = Float32.(inputs), Float32.(targets)
    inputsVal, targetsVal = Float32.(inputsVal), Float32.(targetsVal)
    inputsTest, targetsTest = Float32.(inputsTest), Float32.(targetsTest)

    # Obtener tamaño de la capa de entrada y salida
    numInputs = size(inputs, 2)
    numOutputs = size(targets, 2)

    # Crear la RNA
    ann = buildClassANN(numInputs, topology, numOutputs; transferFunctions=transferFunctions)
    
    # Definir la función de pérdida
    loss(x, y) = Flux.Losses.mse(ann(x), y)

    # Inicializar historial de pérdidas
    lossHistory = Float32[loss(inputs', targets')]
    valLossHistory = isempty(inputsVal) ? Float32[] : Float32[loss(inputsVal', targetsVal')]
    testLossHistory = isempty(inputsTest) ? Float32[] : Float32[loss(inputsTest', targetsTest')]

    # Inicializar el mejor modelo
    bestAnn = deepcopy(ann)
    bestValLoss = isempty(inputsVal) ? Inf32 : valLossHistory[1]
    epochsWithoutImprovement = 0

    # Definir el optimizador
    opt = Flux.Optimise.Adam(learningRate)

    # Obtener los parámetros de la red
    ps = Flux.params(ann)

    # Bucle de entrenamiento
    for epoch in 1:maxEpochs
        Flux.train!(loss, ps, [(inputs', targets')], opt)

        # Calcular pérdidas
        trainLoss = loss(inputs', targets')
        push!(lossHistory, trainLoss)

        valLoss = isempty(inputsVal) ? Inf32 : loss(inputsVal', targetsVal')
        push!(valLossHistory, valLoss)

        testLoss = isempty(inputsTest) ? Inf32 : loss(inputsTest', targetsTest')
        push!(testLossHistory, testLoss)

        # Actualizar la mejor RNA basada en la pérdida de validación
        if !isempty(inputsVal)
            if valLoss < bestValLoss
                bestValLoss = valLoss
                bestAnn = deepcopy(ann)
                epochsWithoutImprovement = 0
            else
                epochsWithoutImprovement += 1
            end

            # Parada temprana
            if epochsWithoutImprovement >= maxEpochsVal
                break
            end
        end

        # Parada temprana si la pérdida es suficientemente baja
        if trainLoss < minLoss
            break
        end
    end

    # Devolver la mejor RNA y los historiales de pérdida
    return (bestAnn, lossHistory, valLossHistory, testLossHistory)
end;


function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)

    
end;