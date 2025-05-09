using CSV, DataFrames, Plots
default(fmt = :png, dpi = 300)

# Leemos los archivos
resultados = CSV.read("resultados/resultados_todas_configuraciones.csv", DataFrame)
mejores = CSV.read("resultados/mejores_configuraciones.csv", DataFrame)

# ========== 1. Accuracy medio por mejor modelo ==========
bar(
    mejores.Modelo,
    mejores.AccuracyMedia,
    yerr = mejores.AccuracyStd,
    title = "Accuracy medio por mejor configuración",
    xlabel = "Modelo",
    ylabel = "Accuracy",
    legend = false,
    rotation = 15
)
savefig("graficas/accuracy_mejores_modelos.png")

# ========== 2. Accuracy por configuración (con etiquetas de hiperparámetros) ==========

function formatear_parametros(param_str)
    param_str = replace(param_str, "=>" => ":")
    param_str = replace(param_str, "Dict(" => "")
    param_str = replace(param_str, ")" => "")
    param_str = replace(param_str, "\"" => "")
    param_str = replace(param_str, r"\s+" => "", count = 0)
    return param_str
end

modelos = unique(resultados.Modelo)

for modelo in modelos
    subdf = filter(:Modelo => m -> m == modelo, resultados)
    
    etiquetas = formatear_parametros.(subdf.Parametros)
    
    plot(
        etiquetas,
        subdf.AccuracyMedia,
        yerr = subdf.AccuracyStd,
        title = "Accuracy por configuración - $modelo",
        xlabel = "Configuración",
        ylabel = "Accuracy",
        legend = false,
        xticks = (1:length(etiquetas), etiquetas),
        rotation = 45,
        seriestype = :bar,
        size=(1000,500)
    )
    
    savefig("graficas/accuracy_$modelo.png")
end


println("Gráficos generados correctamente")