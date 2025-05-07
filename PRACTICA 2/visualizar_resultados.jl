using CSV, DataFrames, Statistics
using StatsPlots  # Requiere también Plots

default(titlefontsize=12, guidefontsize=10, legendfontsize=8)

# 1. Leer resultados
if !isfile("resultados.csv")
    error("❌ No se encontró el archivo 'resultados.csv'. ¿Se ejecutó entrenar_modelos.jl?")
end

df = CSV.read("resultados.csv", DataFrame)

# 2. Cambiar nombre de columnas
rename!(df, "Modelo" => :Model, "AccuracyMedia" => :MeanAcc, "AccuracyStd" => :StdAcc)

# 3. Eliminar duplicados (por si acaso)
unique!(df)

# 4. Extraer el mejor resultado por modelo
mejores = combine(groupby(df, :Model)) do subdf
    subdf[argmax(subdf.MeanAcc), :]
end

# 5. Guardar como tabla CSV
CSV.write("mejores_resultados.csv", mejores)

# 6. Gráfico de barras con desviación típica
@df mejores bar(:Model, :MeanAcc; yerror=:StdAcc,
    title = "Accuracy media por modelo (con desviación típica)",
    xlabel = "Modelo", ylabel = "Accuracy media",
    legend = false, size=(900, 500), bar_width=0.5)
savefig("grafica_accuracy_barra.png")

# 7. Boxplot + scatter de todas las configuraciones
@df df boxplot(:Model, :MeanAcc; label="Boxplot", alpha=0.6, width=0.4)
@df df scatter!(:Model, :MeanAcc; label="Puntos", color=:black, markersize=4)
title!("Distribución de accuracy por modelo y configuración")
xlabel!("Modelo")
ylabel!("Accuracy media")
savefig("grafica_accuracy_distribucion.png")

println("✅ Se han generado las siguientes salidas:")
println("1. grafica_accuracy_barra.png")
println("2. grafica_accuracy_distribucion.png")
println("3. mejores_resultados.csv")
