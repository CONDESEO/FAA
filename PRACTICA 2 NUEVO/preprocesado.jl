using CSV, DataFrames
using StatsBase


df = CSV.read("atp_tennis.csv", DataFrame)

println(names(df))


col = df[!, Symbol("Series")]
frecuencias = countmap(col)
println(frecuencias)
total = sum(values(frecuencias))
porcentajes = Dict(k => 100 * v / total for (k, v) in frecuencias)

#La columna Date se eliminará ya que al usar validación cruzada mezclaremos partidos de fechas muy dispares
#La columna Best Of llevará un preprocesado LABEL ENCODING ya que solo presenta dos valores, 0 o 1
#La columna Tournament se eliminará ya que existen un montón de tipos distintos y puede introducir ruido en el modelo