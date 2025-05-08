using CSV, DataFrames, StatsBase
using MLJ
using MLJModels
using DataFrames
using Statistics
import Pkg

# 1. Leer el CSV
df = CSV.read("atp_tennis.csv", DataFrame)

#X si las moscas
df = unique(df) 

# 3. Filtrar cuotas que no son números válidos (evita errores de parseo)
df = filter(row -> occursin(r"^\d+(\.\d+)?$", row.Odd_1) && occursin(r"^\d+(\.\d+)?$", row.Odd_2), df)

# 4. Convertir cuotas a Float64
df[!, :Odd_1] = parse.(Float64, df.Odd_1)
df[!, :Odd_2] = parse.(Float64, df.Odd_2)

# 5. Eliminar columnas que no se van a usar
select!(df, Not([:Date, :Tournament, :Score, :Rank_1, :Rank_2]))

# 2. Renombrar columnas con espacios para evitar errores
rename!(df, "Best of" => :Best_of)

# 6. Convertir texto a minúsculas
for col in [:Surface, :Court, :Series, :Round]
    df[!, col] = lowercase.(df[!, col])
end

# 7. Agrupar rondas en 3 bloques
function agrupar_round(r)
    if r in ["1st round", "2nd round"]
        return "early"
    elseif r in ["3rd round", "4th round"]
        return "middle"
    elseif r in ["quarterfinals", "semifinals", "the final"]
        return "late"
    else
        return "other"
    end
end

df[!, :RoundGroup] = agrupar_round.(df.Round)
select!(df, Not(:Round))

# 8. Filtrar filas inválidas
df = filter(row ->
    row.Odd_1 > 1 && row.Odd_1 < 50 &&
    row.Odd_2 > 1 && row.Odd_2 < 50, df)

# 9. Codificar Court y Best_of
df[!, :Court] = ifelse.(df.Court .== "indoor", 1, 0)
df[!, :Best_of] = ifelse.(df.Best_of .== 5, 1, 0)

# 10. Crear variables derivadas
df[!, :Odd_Diff] = df.Odd_1 - df.Odd_2
df[!, :Pts_Diff_Rel] = [(p1 + p2) == 0 ? 0.0 : (p1 - p2) / (p1 + p2) for (p1, p2) in zip(df.Pts_1, df.Pts_2)]

# 11. Eliminar columnas originales
select!(df, Not([:Pts_1, :Pts_2, :Odd_1, :Odd_2]))

# 12. Determinar jugadores más frecuentes (top 20)
todos_jugadores = vcat(df.Player_1, df.Player_2)
println(typeof(todos_jugadores))
frecuencias = countmap(todos_jugadores)
jugadores_top20 = first(sort(collect(frecuencias), by = x -> -x[2]), 20)
jugador_dict = Dict(nombre => id for (id, (nombre, _)) in enumerate(jugadores_top20))

# 13. Codificar la columna objetivo Winner ANTES de codificar jugadores
df[!, :Winner] = ifelse.(df.Winner .== df.Player_1, 1, 0)

# 14. Codificar jugadores
df[!, :Player_1] = get.(Ref(jugador_dict), df.Player_1, -1)
df[!, :Player_2] = get.(Ref(jugador_dict), df.Player_2, -1)

# 15. Filtrar solo partidos entre jugadores del top 20
df = filter(row -> row.Player_1 != -1 && row.Player_2 != -1, df)

# 16. Comprobación final
println("Filas finales: ", nrow(df))
println("Distribución de clases: ", countmap(df.Winner))
println(names(df))

CSV.write("atp_tennis_preprocesado.csv", df)

#La columna Date se eliminará ya que al usar validación cruzada mezclaremos partidos de fechas muy dispares
#La columna Best Of llevará un preprocesado LABEL ENCODING ya que solo presenta dos valores, 0 o 1
#La columna Tournament se eliminará ya que existen un montón de tipos distintos y puede introducir ruido en el modelo
#La columna Series llevará un preprocesado One Hot ya que es una variable categórica nominal
#La columna Court llevará un preprocesado LABEL ENCODING ya que solo presenta dos valores, 0 o 1
#La columna Surface llevará un preprocesado One Hot ya que es una variable categórica nominal
#Las columnas Player_1 y Player_2 llevarán un preprocesado LABEL ENCODING ya que existen un montón de valores distintos

#Las columnas Pts_1 y Pts_2 que almacenan los puntos de los jugadores (1, 2) deberán ser normalizados para 
    #modelos sensibles a escala
    

#La columna round tiene desbalance, HACER ESTO: AGRUPAR EN  3 BLOQUES(INICIAL, INTERMEDIO, FINAL)  
    # para reducir el desequilibrio mejorar la generalización del modelo y
    # conservar el significado competitivo.
    
#En las colimnas de Ods vamos a convertir las cadenas de texto a números(float64), eliminar los puntos de los números,
     # eliminar las filas en las que no existe valor de apuesta por fecha (-1), eliminar las mayores a 50 y después en cada fold se normalizará con z-score