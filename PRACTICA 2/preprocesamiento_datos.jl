include("modelo_utils.jl")  # oneHotEncoding, normalizeMinMax! deben estar aquí

using CSV, DataFrames

# -------------------------------
# Cargar CSV
# -------------------------------
df = CSV.read("atp_tennis.csv", DataFrame)

# -------------------------------
# Selección de columnas
# -------------------------------
keep_cols = ["Series", "Court", "Surface", "Round", "Best of",
             "Player_1", "Player_2", "Rank_1", "Rank_2",
             "Pts_1", "Pts_2", "Odd_1", "Odd_2", "Winner"]
df = df[:, keep_cols]

# -------------------------------
# Filtrar por los 100 jugadores más frecuentes
# -------------------------------
all_players = vcat(df.Player_1, df.Player_2)
top_players = sort(collect(countmap(all_players)), by=x -> -x[2])[1:10]
top_players_set = Set(p[1] for p in top_players)

df = filter(row -> row.Player_1 in top_players_set && row.Player_2 in top_players_set, df)

# -------------------------------
# Reparar odds inválidas
# -------------------------------
for col in ["Odd_1", "Odd_2"]
    df[!, col] = map(x -> try
            val = parse(Float64, string(x))
            (val < 0.0 || val > 100.0) ? 1.0 : val
        catch
            1.0
        end, df[!, col])
end

# -------------------------------
# Guardar etiquetas originales
# -------------------------------
raw_targets = df.Winner
select!(df, Not(:Winner))  # Eliminar la columna objetivo del DataFrame

# -------------------------------
# Convertir columnas a string (para one-hot)
# -------------------------------
for col in names(df)
    df[!, col] = string.(df[!, col])
end

# -------------------------------
# Codificar entradas
# -------------------------------
encoded_inputs = hcat([oneHotEncoding(df[!, col]) for col in names(df)]...)
inputs = Float32.(encoded_inputs)

# -------------------------------
# Codificar salidas
# -------------------------------
clases = unique(raw_targets)
targets = oneHotEncoding(raw_targets, clases)

# -------------------------------
# Normalizar las columnas numéricas
# Las 6 últimas: Rank_1, Rank_2, Pts_1, Pts_2, Odd_1, Odd_2
num_cols = 6
normalizeMinMax!(inputs[:, end - num_cols + 1:end])

# -------------------------------
# Confirmación
# -------------------------------
println("✔️ Preprocesamiento completo")
println("Tamaño entradas: ", size(inputs))
println("Tamaño salidas: ", size(targets))
