using CSV, DataFrames, Statistics, Random
using MLJ, LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface

include("modelUtils.jl")
include("preprocesado.jl")

println(df.Winner)
println(typeof(df.Winner))
# Definir las columnas para One-Hot Encoding
cols_onehot = [:Series, :Surface, :RoundGroup]
#Definir las columnas para Normalización
cols_normalize = [:Odd_Diff, :Pts_Diff_Rel]

# Número de pliegues para cross-validation
k = 5

# Realizar cross-validation
folds = crossvalidation(nrow(df), k)  # Obtener los pliegues

for fold in 1:k
    # Dividir los datos en conjunto de entrenamiento y conjunto de prueba
    train_indices = findall(x -> x != fold, folds)
    test_indices = findall(x -> x == fold, folds)

    X_train = df[train_indices, :]  # Conjunto de entrenamiento
    X_test = df[test_indices, :]  # Conjunto de prueba
    y_train = X_train.Winner  # Etiquetas de entrenamiento
    y_test = X_test.Winner  # Etiquetas de prueba

    # Aplicar One-Hot Encoding solo a los datos de entrenamiento
    for col in cols_onehot
        classes = unique(X_train[!, col])  # Obtener las categorías únicas del conjunto de entrenamiento
        X_train_encoded = oneHotEncoding(X_train[!, col], classes)  # Realizar One-Hot Encoding

        # Añadir las nuevas columnas One-Hot codificadas a X_train
        for (i, class) in enumerate(classes)
            new_col_name = Symbol(string(col, "_", class))  # Crear el nombre de la nueva columna
            X_train[!, new_col_name] .= X_train_encoded[:, i]  # Añadir la columna One-Hot codificada
        end
    end

    #Aplicar One-Hot Encoding a los datos de prueba
    for col in cols_onehot
        classes = unique(X_train[!, col])  # Obtener las categorías únicas del conjunto de entrenamiento
        X_test_encoded = oneHotEncoding(X_test[!, col], classes)  # Realizar One-Hot Encoding

        # Añadir las nuevas columnas One-Hot codificadas a X_test
        for (i, class) in enumerate(classes)
            new_col_name = Symbol(string(col, "_", class))  # Crear el nombre de la nueva columna
            X_test[!, new_col_name] .= X_test_encoded[:, i]  # Añadir la columna One-Hot codificada
        end
    end

    # Aplicar Normalización a los datos de entrenamiento
    for col in cols_normalize
      # Extraer la columna a normalizar del conjunto de entrenamiento
      train_column = X_train[!, col]
    
      # Convertir el vector a una matriz (para cumplir con el tipo requerido por la función)
      train_column_matrix = reshape(train_column, :, 1)
    
      # Imprimir estadísticas antes de la normalización
      println("Antes de la normalización - Entrenamiento: ", col)
      println("Media: ", mean(train_column_matrix))
      println("Desviación estándar: ", std(train_column_matrix))
    
      # Calcular los parámetros de normalización (mínimo y máximo o media y desviación estándar)
      normalization_params = calculateZeroMeanNormalizationParameters(train_column_matrix)
    
      # Aplicar normalización en el conjunto de entrenamiento
      normalizeZeroMean!(train_column_matrix, normalization_params)
    
      # Imprimir estadísticas después de la normalización
      println("Después de la normalización - Entrenamiento: ", col)
      println("Media: ", mean(train_column_matrix))
      println("Desviación estándar: ", std(train_column_matrix))
    
      # Guardar los valores normalizados de nuevo en el dataframe
      X_train[!, col] .= train_column_matrix[:, 1]  # Convertir de nuevo a vector para el dataframe
    end

    # Aplicar Normalización a los datos de prueba
    for col in cols_normalize
      # Extraer la columna a normalizar del conjunto de entrenamiento
      test_column = X_test[!, col]
    
      # Convertir el vector a una matriz (para cumplir con el tipo requerido por la función)
      test_column_matrix = reshape(test_column, :, 1)
    
      # Imprimir estadísticas antes de la normalización
      println("Antes de la normalización - Entrenamiento: ", col)
      println("Media: ", mean(test_column_matrix))
      println("Desviación estándar: ", std(test_column_matrix))
    
      # Calcular los parámetros de normalización (mínimo y máximo o media y desviación estándar)
      normalization_params = calculateZeroMeanNormalizationParameters(test_column_matrix)
    
      # Aplicar normalización en el conjunto de entrenamiento
      normalizeZeroMean!(test_column_matrix, normalization_params)
    
      # Imprimir estadísticas después de la normalización
      println("Después de la normalización - Prueba: ", col)
      println("Media: ", mean(test_column_matrix))
      println("Desviación estándar: ", std(test_column_matrix))
    
      # Guardar los valores normalizados de nuevo en el dataframe
      X_test[!, col] .= test_column_matrix[:, 1]  # Convertir de nuevo a vector para el dataframe
    end
    

    


end

