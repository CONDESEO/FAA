

# Archivo de pruebas para realizar autoevaluación de algunas funciones de los ejercicios

# Importamos el archivo con las soluciones a los ejercicios
include("ejercicio2.jl");
#   Cambiar "soluciones.jl" por el nombre del archivo que contenga las funciones desarrolladas



# Fichero de pruebas realizado con la versión 1.11.2 de Julia
println(VERSION)
#  y la 1.11.3 de Random
println(Random.VERSION)
#  y la versión 0.14.25 de Flux
import Pkg
Pkg.status("Flux")

# Es posible que con otras versiones los resultados sean distintos, estando las funciones bien, sobre todo en la funciones que implican alguna componente aleatoria




# Cargamos el dataset
using DelimitedFiles: readdlm
dataset = readdlm("iris.data",',');
# Preparamos las entradas
inputs = convert(Array{Float32,2}, dataset[:,1:4]);


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------


# Hacemos un one-hot-encoding a las salidas deseadas
targets = oneHotEncoding(dataset[:,5]);
# Nos aseguramos de que la matriz de salidas deseadas tiene valores correctos
@assert(size(targets)==(150,3))
@assert(all(targets[  1:50 ,1]) && !any(targets[  1:50,  2:3 ])); # Primera clase
@assert(all(targets[ 51:100,2]) && !any(targets[ 51:100,[1,3]])); # Segunda clase
@assert(all(targets[101:150,3]) && !any(targets[101:150, 1:2] )); # Tercera clase



# Comprobamos que las funciones de normalizar funcionan correctamente
# Normalizacion entre maximo y minimo
newInputs = normalizeMinMax(inputs);
@assert(!isnothing(newInputs))
@assert(all(minimum(newInputs, dims=1) .== 0));
@assert(all(maximum(newInputs, dims=1) .== 1));
# Normalizacion de media 0. en este caso, debido a redondeos, la media y desviacion tipica de cada variable no van a dar exactamente 0 y 1 respectivamente. Por eso las comprobaciones se hacen de esta manera
newInputs = normalizeZeroMean(inputs);
@assert(!isnothing(newInputs))
@assert(all(abs.(mean(newInputs, dims=1)) .<= 1e-4));
@assert(all(isapprox.(std( newInputs, dims=1), 1)));

# Finalmente, normalizamos las entradas entre maximo y minimo:
normalizeMinMax!(inputs);


# Probamos la función classifyOutputs:
@assert(classifyOutputs(0.1:0.1:1; threshold=0.65) == [falses(6); trues(4)]);
@assert(classifyOutputs([1 2 3; 3 2 1; 2 3 1; 2 1 3]) == Bool[0 0 1; 1 0 0; 0 1 0; 0 0 1]);


# Comprobamos que la creación de la RNA funciona correctamente:
ann = buildClassANN(4, [5], 3);
@assert(length(ann)==3)
@assert(ann[3]==softmax)
@assert(size(ann[1].weight)==(5,4))
@assert(size(ann[2].weight)==(3,5))
@assert(size(ann(inputs'))==(3,150))

# Comprobamos que la función accuracy funciona correctamente:
@assert(isapprox(accuracy([sin.(1:150) cos.(1:150) tan.(1:150)], targets), 0.34))
@assert(isapprox(accuracy(1:(-1/150):(1/150), targets[:,1]; threshold=0.75), 0.92))