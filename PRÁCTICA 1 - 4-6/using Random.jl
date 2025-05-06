using Random
using Statistics
using Flux
using Flux.Losses
using SymDoME

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    # Comprobación opcional: todos los valores de feature deben estar en classes
    if any(x -> !(x in classes), feature)
        error("Todos los elementos de 'feature' deben estar en 'classes'")
    end

    # Codificación one-hot genérica para cualquier número de clases
    oneHotMatrix = convert(BitArray{2}, hcat([instance .== classes for instance in feature]...)')
    return oneHotMatrix
end

function oneHotEncoding(feature::AbstractArray{<:Any,1})
    classes = unique(feature)
    return oneHotEncoding(feature, classes)
end

function oneHotEncoding(feature::AbstractArray{Bool,1})
    return hcat(.!feature, feature) .|> Int
end

#//////////////////////////////////////////////////////////////////////////////////////////////////////

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return (
        min = minimum(dataset, dims=1),
        max = maximum(dataset, dims=1)
    )
end

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return (
        mean = mean(dataset, dims=1),
        std = std(dataset, dims=1)
    )
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, params::NamedTuple{(:min, :max), Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Real,2}}}
)
    dataset .= (dataset .- params.min) ./ (params.max .- params.min)
    dataset[:, vec(params.min .== params.max)] .= 0
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    normalizeMinMax!(dataset, normalizationParameters)
    return normalizationParameters
end

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}
)
    minValues, maxValues = normalizationParameters
    normalizedDataset = copy(dataset)
    normalizedDataset .= (normalizedDataset .- minValues) ./ (maxValues .- minValues)
    normalizedDataset[:, vec(minValues .== maxValues)] .= 0
    return normalizedDataset
end

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    return normalizeMinMax(dataset, normalizationParameters)
end

function normalizeZeroMean!(
    dataset::AbstractArray{<:Real,2},
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}
)
    meanValues, stdValues = normalizationParameters
    dataset .= (dataset .- meanValues) ./ stdValues
    dataset[:, vec(stdValues .== 0)] .= 0
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset)
    normalizeZeroMean!(dataset, normalizationParameters)
end

function normalizeZeroMean(
    dataset::AbstractArray{<:Real,2},
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}
)
    meanValues, stdValues = normalizationParameters
    normalizedDataset = copy(dataset)
    normalizedDataset .= (normalizedDataset .- meanValues) ./ stdValues
    normalizedDataset[:, vec(stdValues .== 0)] .= 0
    return normalizedDataset
end

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset)
    return normalizeZeroMean(dataset, normalizationParameters)
end

#///////////////////////////////////////////////////////////////////////////////////////

function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    result = outputs .>= threshold
    return result
end

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    numCols = size(outputs, 2)

    if numCols == 1
        result = classifyOutputs(outputs[:]; threshold=threshold)
        return reshape(result, :, 1)
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2)
        classified_outputs = falses(size(outputs))
        
        # Generar índice de fila: 1, 2, 3, ..., n
        row_indices = collect(1:size(outputs, 1))

        # Usar indexación vectorizada
        classified_outputs[CartesianIndex.(row_indices, vec(indicesMaxEachInstance))] .= true

        return classified_outputs
    end
end

#/////////////////////////////////////////////////////////////////////////////////////////////////////////

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))

    ann = Chain()
    numInputsLayer = numInputs

    if !isempty(topology)
        for (i, numOutputsLayer) in enumerate(topology)
            layer = Dense(numInputsLayer, numOutputsLayer, transferFunctions[i])
            ann = Chain(ann..., layer)
            numInputsLayer = numOutputsLayer
        end
    end

    if numOutputs > 2
        outputLayer = Dense(numInputsLayer, numOutputs, identity)
        ann = Chain(ann..., outputLayer, softmax)
    else
        outputLayer = Dense(numInputsLayer, numOutputs, σ)
        ann = Chain(ann..., outputLayer)
    end

    return ann
end

#///////////////////////////////////////////////////////////////////////////////////////////////

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    correct = outputs .== targets
    return mean(correct)
end

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    numCols = size(outputs, 2)

    if numCols == 1
        return accuracy(outputs[:,1], targets[:,1])
    else
        comparisonMatrix = outputs .== targets 
        correctRows = all(comparisonMatrix, dims=2)
        return mean(correctRows)
    end
end

