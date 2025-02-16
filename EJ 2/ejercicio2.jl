using Statistics
using Flux
using Flux.Losses


#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


using DelimitedFiles

# Cargamos el dataset
dataset = readdlm("iris.data",',');

# Preparamos las entradas
inputs = dataset[:,1:4];
# Con cualquiera de estas 3 maneras podemos convertir la matriz de entradas de tipo Array{Any,2} en Array{Float32,2}, si los valores son numéricos:
inputs = Float32.(inputs);
inputs = convert(Array{Float32,2},inputs);
inputs = [Float32(x) for x in inputs];
println("Tamaño de la matriz de entradas: ", size(inputs,1), "x", size(inputs,2), " de tipo ", typeof(inputs));

# Preparamos las salidas deseadas codificándolas puesto que son categóricas
targets = dataset[:,5];
println("Longitud del vector de salidas deseadas antes de codificar: ", length(targets), " de tipo ", typeof(targets));
classes = unique(targets);
numClasses = length(classes);


#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    numClasses = length(classes)
    numSamples = length(feature)
    
    if numClasses <= 2
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
            oneHotMatrix[i, 2] = 1  # True -> [0, 1]
        else
            oneHotMatrix[i, 1] = 1  # False -> [1, 0]
        end
    end
    
    return oneHotMatrix
end;


#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////


function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    MinMaxParameters = (minimum(input, dim=1), maximum(input, dim=1))
    return MinMaxParameters
end;

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    ZeroMeanParameters = (dataset(input, dim=1), dataset(input, dim=1))
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

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues, maxValues = normalizationParameters
    normalizedDataset = (dataset .-minValues) ./ (maxValues .- minValues)
    normalizedDataset[:, vec(minValues .== maxValues)] .= 0
    return normalizedDataset
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
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
    normalizedDataset = (dataset .- meanValues) ./ std.Values
    normalizedDataset[:, vec(stdValues .== 0)] .= 0
    return normalizedDataset
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    normalizedParameters = calculateZeroMeanNormalizationParameters(dataset)
    return normalizeZeroMean(dataset, normalizedParameters)
end;


#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    classifiedOutputs = outputs .>= threshold
    return classifiedOutputs
end;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    classifiedOutputs = outputs .>= threshold
    return classifiedOutputs
end;


#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////


function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert size(outputs) == size(targets) "Las dimensiones de outputs y targets deben coincidir"
    correctPredictions = sum(outputs .== targets)
    totalPredictions = length(outputs)
    return correctPredictions / totalPredictions
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    @assert size(outputs) == size(targets) "Las dimensiones de outputs y targets deben coincidir"
    correctPredictions = sum(all(outputs .== targets, dims=2))
    totalPredictions = size(outputs, 1) 
    return totalPredictions > 0 ? correctPredictions[1] / totalPredictions : 0.0
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    @assert size(outputs) == size(targets) "Las dimensiones de outputs y targets deben coincidir"

    classifiedOutputs = outputs .>= threshold  # Convierte a booleano según threshold
    correctPredictions = sum(classifiedOutputs .== targets)
    totalPredictions = length(outputs)

    return correctPredictions / totalPredictions
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    @assert size(outputs) == size(targets) "Las dimensiones de outputs y targets deben coincidir"

    classifiedOutputs = outputs >= threshold
    correctPredictions = sum(all(classifiedOutputs .== targets, dims=2))
    totalPredictions = size(targets, 1)

    return totalPredictions > 0 ? correctPredictions[1] / totalPredictions : 0.0
end;


#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////


function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    #
    # Codigo a desarrollar
    #
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    #
    # Codigo a desarrollar
    #
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    #
    # Codigo a desarrollar
    #
end;