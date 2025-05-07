using CSV, DataFrames
using MLJ

# -----------------------------------------
# Función de preprocesamiento general
# -----------------------------------------
function cargar_y_filtrar_datos()
    df = CSV.read("atp_tennis.csv", DataFrame)

    # Convertir columnas numéricas que pueden venir mal tipadas
    df.Odd_1 = parse.(Float64, replace.(string.(df.Odd_1), "." => ""))
    df.Odd_2 = parse.(Float64, replace.(string.(df.Odd_2), "." => ""))

    # Eliminar partidos con datos inválidos
    df = dropmissing(df)
    df = filter(row -> row.Odd_1 > 0 && row.Odd_2 > 0, df)

    # Filtrar jugadores más frecuentes
    jugadores = vcat(df.Player_1, df.Player_2)
    conteo = countmap(jugadores)
    top_jugadores = sort(collect(keys(conteo)), by = x -> -conteo[x])[1:20]
    df = filter(row -> row.Player_1 in top_jugadores && row.Player_2 in top_jugadores, df)

    # Generar diccionario de label encoding
    player_dict = Dict(j => i for (i, j) in enumerate(top_jugadores))

    # El target será uno de los jugadores (ganador)
    clases = sort(collect(keys(player_dict)))

    return df, player_dict, clases
end

# -----------------------------------------
# Función de preprocesamiento por fold
# -----------------------------------------
function procesar_fold(
    df_fold::DataFrame,
    player_dict::Dict{String, Int},
    clases::Vector{String};
    one_hot::Bool = false
)
    # Label encoding para los jugadores
    df_fold.Player_1 = map(p -> player_dict[string(p)], df_fold.Player_1)
    df_fold.Player_2 = map(p -> player_dict[string(p)], df_fold.Player_2)

    # Label encoding del ganador
    y = map(w -> player_dict[string(w)], df_fold.Winner)

    # Selección de variables predictoras
    X = select(df_fold, Not(["Winner"]))
    X_encoded = MLJ.table(X)

    if one_hot
        y_encoded = oneHotEncoding(y, collect(1:length(player_dict)))
    else
        y_encoded = y
    end

    return Matrix(X_encoded), y_encoded
end


# -----------------------------------------
# Cargar los datos una vez
# -----------------------------------------

global df_filtrado, player_dict, clases = cargar_y_filtrar_datos()
global raw_targets = df_filtrado.Winner

println("✔️ Preprocesamiento inicial completo. Listo para procesar folds.")
