using CSV
using DataFrames
include("funciones.jl")  # Asegúrate de tener oneHotEncoding y normalizeMinMax! definidos ahí

# -------------------------------
# Cargar el dataset
# -------------------------------
df = CSV.read("atp_tennis.csv", DataFrame)

# Eliminar columnas que no se usarán
select!(df, Not(["Tournament", "Date", "Player_1", "Player_2", "Score"]))

# -------------------------------
# Definir columnas categóricas y numéricas
# -------------------------------
cat_cols = ["Series", "Court", "Surface", "Round", "Best of"]
num_cols = ["Rank_1", "Rank_2", "Pts_1", "Pts_2", "Odd_1", "Odd_2"]
target_col = "Winner"

# -------------------------------
# Arreglar valores anómalos en Odds
# -------------------------------
for col in ["Odd_1", "Odd_2"]
    df[!, col] = map(x -> try
        val = parse(Float32, string(x))
        (val < 0 || val > 100) ? 1.0f0 : val
    catch
        1.0f0
    end, df[!, col])
end

# -------------------------------
# Extraer targets y eliminarlos del dataframe
# -------------------------------
raw_targets = df[!, target_col]
select!(df, Not(target_col))

# -------------------------------
# Codificar categóricas
# -------------------------------
encoded_cats = hcat([oneHotEncoding(df[!, col]) for col in cat_cols]...)

# -------------------------------
# Procesar numéricas (parsear y stackear)
# -------------------------------
function safe_parse_num(col)
    return map(x -> try
        parse(Float32, string(x))
    catch
        NaN32
    end, col)
end

numeric_data = hcat([safe_parse_num(df[!, col]) for col in num_cols]...)

# Normalizamos
normalizeMinMax!(numeric_data)

# -------------------------------
# Concatenar todo y convertir
# -------------------------------
inputs = Float32.(hcat(encoded_cats, numeric_data))
targets = oneHotEncoding(raw_targets)

# -------------------------------
# Info
# -------------------------------
println("✔️ Preprocesamiento terminado")
println("Tamaño entradas: ", size(inputs))
println("Tamaño salidas: ", size(targets))

println("\n🧩 Variables categóricas codificadas:")
for col in cat_cols
    println(col, " => ", length(unique(df[!, col])))
end

println("\n📈 Primeras 5 filas del dataset (entrada):")
for i in 1:5
    println(inputs[i, :])
end
