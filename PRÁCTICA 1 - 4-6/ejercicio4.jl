# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    precision = sum(outputs .& targets) / sum(outputs)
    error_rate = sum(outputs .!= targets) / length(outputs)
    sensitivity = sum(outputs .& targets) / sum(targets)
    recall = sum((!outputs) .& (!targets)) / sum(!targets)
    positive_predictive_value = sum(outputs .& targets) / sum(outputs)
    negative_predictive_value = sum((!outputs) .& (!targets)) / sum(!outputs)
    F1_score = 2 * precision * sensitivity / (precision + sensitivity)
    
    true_positive = sum(outputs .& targets)
    false_positive = sum(outputs .& (!targets))
    true_negative = sum((!outputs) .& (!targets))
    false_negative = sum((!outputs) .& targets)
    
    confusion_matrix = [true_positive false_positive; false_negative true_negative]

    return (precision, error_rate, sensitivity, recall, positive_predictive_value, negative_predictive_value, F1_score, confusion_matrix)
end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    outputs = outputs .>= threshold
    return confusionMatrix(outputs, targets)
end;

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

using SymDoME


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;
