using CSV, DataFrames, Statistics
using StatsPlots
import StatsPlots: boxplot, boxplot!, stripplot, stripplot!
default(titlefontsize=12, guidefontsize=10, legendfontsize=8)

# 1. Leer resultados
df = CSV.read("resultados.csv", DataFrame)

# 2. Extraer el mejor resultado por modelo (mayor AccuracyMedia)
mejores = combine(groupby(df, :Modelo)) do subdf
    subdf[argmax(subdf.AccuracyMedia), :]
end

# 3. Guardar como tabla CSV
CSV.write("mejores_resultados.csv", mejores)

# 4. Gráfico de barras con desviación típica
@df mejores bar(:Modelo, :AccuracyMedia; yerror=:AccuracyStd,
    title = "Accuracy media por modelo (con desviación típica)",
    xlabel = "Modelo", ylabel = "Accuracy media",
    legend = false, size=(900, 500), bar_width=0.5)
savefig("grafica_accuracy_barra.png")

# 5. Gráfico de distribución por configuraciones (box + puntos)
@df df boxplot(:Modelo, :AccuracyMedia; label="Boxplot", alpha=0.6, width=0.4)
@df df scatter!(:Modelo, :AccuracyMedia; label="Puntos", color=:black, markersize=4, jitter=0.3)

title!("Distribución de accuracy por modelo y configuración")
xlabel!("Modelo")
ylabel!("Accuracy media")
savefig("grafica_accuracy_distribucion.png")

println("✅ Se han generado las siguientes salidas:")
println("1. grafica_accuracy_barra.png")
println("2. grafica_accuracy_distribucion.png")
println("3. mejores_resultados.csv")
