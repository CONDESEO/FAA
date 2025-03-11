# =============================================================================
# ===================== Ejercicio 4 ============================
# =============================================================================

# Versión para vectores booleanos (clasificación binaria)
function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert length(outputs) == length(targets) "outputs y targets deben tener la misma longitud."
    @assert length(outputs) > 0 "Los vectores no pueden estar vacíos."
    
    # Componentes de la matriz de confusión
    TN = sum(.!outputs .& .!targets)  # Verdaderos Negativos
    FN = sum(.!outputs .& targets)     # Falsos Negativos
    TP = sum(outputs .& targets)       # Verdaderos Positivos
    FP = sum(outputs .& .!targets)      # Falsos Positivos
    
    total = TN + TP + FN + FP
    @assert total > 0 "El total de instancias debe ser mayor que cero."
    
    # Matriz de confusión (filas: clase real; columnas: predicción)
    confMatrix = [TN FP;
                  FN TP]
    
    # Métricas
    acc       = (TN + TP) / total
    errorRate = 1 - acc
    # Se evitan divisiones por cero asignando 1 cuando no hay instancias positivas/negativas
    recall      = (TP == 0 && FN == 0) ? 1.0 : TP / (TP + FN)       # Sensibilidad
    specificity = (TN == 0 && FP == 0) ? 1.0 : TN / (TN + FP)       # Especificidad
    precision   = (TP == 0 && FP == 0) ? 1.0 : TP / (TP + FP)       # Valor predictivo positivo (PPV)
    npv         = (TN == 0 && FN == 0) ? 1.0 : TN / (TN + FN)       # Valor predictivo negativo (NPV)
    F1          = (recall == 0 && precision == 0) ? 0.0 : 2 * (recall * precision) / (recall + precision)
    
    @assert !isnan(F1) "F1-score no debe ser NaN."
    
    return (acc, errorRate, recall, specificity, precision, npv, F1, confMatrix)
end

# Versión para vectores de valores reales (se convierte a booleano usando threshold)
function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    bool_outputs = outputs .>= threshold
    return confusionMatrix(bool_outputs, targets)
end

# Versión para matrices booleanas (problema multiclase)
function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert size(outputs) == size(targets) "Las dimensiones de outputs y targets deben coincidir."
    nClasses = size(outputs, 2)
    # Si hay una sola columna, delegamos a la versión vectorial
    if nClasses == 1
        return confusionMatrix(outputs[:,1], targets[:,1])
    end
    @assert nClasses != 2 "Para clasificación binaria use la versión vectorial de confusionMatrix."
    
    # Prealocar vectores para las métricas por clase
    recall_vec      = zeros(Float64, nClasses)
    specificity_vec = zeros(Float64, nClasses)
    precision_vec   = zeros(Float64, nClasses)
    npv_vec         = zeros(Float64, nClasses)
    F1_vec          = zeros(Float64, nClasses)
    
    for i in 1:nClasses
        # Para cada clase, calcular las métricas usando la versión vectorial
        (_, _, r, s, p, n, f, _) = confusionMatrix(outputs[:, i], targets[:, i])
        recall_vec[i]      = r
        specificity_vec[i] = s
        precision_vec[i]   = p
        npv_vec[i]         = n
        F1_vec[i]          = f
    end
    
    # Construir la matriz de confusión multiclasificación:
    # Elemento (i,j) = número de instancias de clase real i clasificadas como j
    confMatrix = [sum(targets[:, i] .& outputs[:, j]) for i in 1:nClasses, j in 1:nClasses]
    
    # Calcular precisión global (accuracy)
    total = sum(confMatrix)
    diag_sum = sum(diag(confMatrix))
    acc = total == 0 ? 0.0 : diag_sum / total
    errorRate = 1 - acc
    
    # Combinar las métricas por clase usando estrategia weighted o macro
    if weighted
        weights = vec(sum(targets, dims=1))
        weights = weights / sum(weights)
        overall_recall      = sum(weights .* recall_vec)
        overall_specificity = sum(weights .* specificity_vec)
        overall_precision   = sum(weights .* precision_vec)
        overall_npv         = sum(weights .* npv_vec)
        overall_F1          = sum(weights .* F1_vec)
    else
        overall_recall      = mean(recall_vec)
        overall_specificity = mean(specificity_vec)
        overall_precision   = mean(precision_vec)
        overall_npv         = mean(npv_vec)
        overall_F1          = mean(F1_vec)
    end
    
    return (acc, errorRate, overall_recall, overall_specificity, overall_precision, overall_npv, overall_F1, confMatrix)
end

# Versión para matrices de valores reales (se convierten a booleanos)
function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    bool_outputs = outputs .>= threshold
    return confusionMatrix(bool_outputs, targets; weighted=weighted)
end

# Versión para vectores de etiquetas (cualquier tipo) recibiendo vector de clases
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert all([in(label, classes) for label in vcat(targets, outputs)]) "Todas las etiquetas deben estar en 'classes'."
    onehot_outputs = oneHotEncoding(outputs, classes)
    onehot_targets = oneHotEncoding(targets, classes)
    return confusionMatrix(onehot_outputs, onehot_targets; weighted=weighted)
end

# Versión para vectores de etiquetas (cualquier tipo) sin especificar clases
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs))
    return confusionMatrix(outputs, targets, classes; weighted=weighted)
end

using SymDoME  # Se asume que el paquete SymDoME está instalado y disponible

# Versión para problemas de clasificación binaria (targets es vector Bool)
function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
                        testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs, trainingTargets = trainingDataset
    # Convertir entradas a Float64 para mayor precisión
    trainingInputs = Float64.(trainingInputs)
    testInputs     = Float64.(testInputs)
    # Entrenar con la función dome; se ignoran los tres primeros valores y se extrae el modelo
    _, _, _, model = dome(trainingInputs, trainingTargets; maximumNodes=maximumNodes)
    testOutputs = evaluateTree(model, testInputs)
    if isa(testOutputs, Real)
        testOutputs = repeat([testOutputs], size(testInputs, 1))
    end
    return testOutputs  # Vector de Float64
end

# Versión para clasificación multiclase con targets como matriz Bool (cada columna: una clase)
function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
                        testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs, trainingTargets = trainingDataset
    nClasses = size(trainingTargets, 2)
    # Si hay una sola columna, se delega a la versión binaria
    if nClasses == 1
        testOutputs_vec = trainClassDoME((trainingInputs, vec(trainingTargets)), testInputs, maximumNodes)
        return reshape(testOutputs_vec, :, 1)
    end
    @assert nClasses != 2 "Para clasificación multiclase, el número de columnas debe ser distinto de 2."
    
    nTest = size(testInputs, 1)
    testOutputs = Array{Float64}(undef, nTest, nClasses)
    # Estrategia "uno contra todos": entrenar un clasificador para cada clase (cada columna)
    for i in 1:nClasses
        targets_i = trainingTargets[:, i]  # Vector Bool para la i-ésima clase
        testOutputs[:, i] = trainClassDoME((trainingInputs, targets_i), testInputs, maximumNodes)
    end
    return testOutputs  # Matriz de Float64 (una columna por clase)
end

# Versión para clasificación con etiquetas de cualquier tipo (targets es vector de Any)
function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
                        testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs, trainingTargets = trainingDataset
    # Determinar las clases únicas a partir de los targets
    classes = unique(trainingTargets)
    # Convertir los targets a codificación one-hot (matriz Bool)
    onehot_targets = oneHotEncoding(trainingTargets, classes)
    # Llamar a la versión de trainClassDoME para clasificación multiclase (targets Bool)
    testOutputsDoME = trainClassDoME((trainingInputs, onehot_targets), testInputs, maximumNodes)
    # testOutputsDoME es una matriz de Float64 (una columna por clase)
    # Convertir estas salidas a clasificación booleana usando classifyOutputs (umbral = 0)
    testOutputsBool = classifyOutputs(testOutputsDoME; threshold=0)
    nTest = size(testInputs, 1)
    # Prealocar vector de salida con el mismo tipo que las etiquetas originales
    testOutputs_final = Array{eltype(trainingTargets),1}(undef, nTest)
    if length(classes) <= 2
        testOutputsBool = vec(testOutputsBool)
        testOutputs_final[testOutputsBool] .= classes[1]
        if length(classes) == 2
            testOutputs_final[.!testOutputsBool] .= classes[2]
        end
    else
        for i in 1:length(classes)
            testOutputs_final[testOutputsBool[:, i]] .= classes[i]
        end
    end
    return testOutputs_final  # Vector con etiquetas (del mismo tipo que trainingTargets)
end

# Versión para matrices booleanas (multiclase)
function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    (acc, errorRate, recall, specificity, precision, npv, F1, confMatrix) = confusionMatrix(outputs, targets; weighted=weighted)
    println("Matriz de Confusión:")
    nClasses = size(confMatrix, 1)
    for i in 1:nClasses
        println(confMatrix[i, :])
    end
    println("Accuracy: ", acc)
    println("Error Rate: ", errorRate)
    println("Recall: ", recall)
    println("Specificity: ", specificity)
    println("Precision: ", precision)
    println("NPV: ", npv)
    println("F1-score: ", F1)
    return (acc, errorRate, recall, specificity, precision, npv, F1, confMatrix)
end

# Versión para matrices de valores reales (se convierten a booleanos usando threshold=0.5 por defecto)
function printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    bool_outputs = outputs .>= 0.5
    return printConfusionMatrix(bool_outputs, targets; weighted=weighted)
end

# Versión para vectores de etiquetas (cualquier tipo) recibiendo vector de clases
function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1},
                              classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    (acc, errorRate, recall, specificity, precision, npv, F1, confMatrix) = confusionMatrix(outputs, targets, classes; weighted=weighted)
    println("Matriz de Confusión:")
    nClasses = size(confMatrix, 1)
    for i in 1:nClasses
        println(confMatrix[i, :])
    end
    println("Accuracy: ", acc)
    println("Error Rate: ", errorRate)
    println("Recall: ", recall)
    println("Specificity: ", specificity)
    println("Precision: ", precision)
    println("NPV: ", npv)
    println("F1-score: ", F1)
    return (acc, errorRate, recall, specificity, precision, npv, F1, confMatrix)
end

# Versión para vectores de etiquetas (cualquier tipo) sin especificar clases
function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs))
    return printConfusionMatrix(outputs, targets, classes; weighted=weighted)
end
