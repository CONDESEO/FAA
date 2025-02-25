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
    #
    # Codigo a desarrollar
    #
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end;