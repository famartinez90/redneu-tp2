# -*- coding: utf-8 -*-
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import self_organized_map as som
import parameters as params
import encoder as encoder


######### PARSEO DE PARAMETROS ##############

filepath, eta, epochs, regla, dim_salida, red_desde_archivo, red_hacia_archivo, red_ej1 = params.iniciar()

######### PARSEO DE DATOS ##############

f = open('tp2_training_dataset.csv', 'rb')
reader = csv.reader(f)
categorias_verificacion = []
atributos = []

for row in reader:
    categoria = float(row.pop(0))
    categorias_verificacion.append(categoria)
    atributos.append([float(x) for x in row])

f.close()

######## MEDIA 0 POR COLUMNA ##############

matrix = np.array(atributos)
column_means = np.mean(matrix, axis=0)

for i, mean in enumerate(column_means):
    for j, _ in enumerate(matrix):
        matrix[j][i] = matrix[j][i] - mean

# Aca ordenamos al azar los documentos y sus categorias
# de manera de siempre agarrar distintos conjuntos de train y validation
rnd_state = np.random.get_state()
np.random.shuffle(matrix)
np.random.set_state(rnd_state)
np.random.shuffle(categorias_verificacion)


dataset_train = matrix[:int(len(matrix) * 0.9)]
dataset_validation = matrix[int(len(matrix) * 0.9):]

######## TRAINING ##############

n_entrada = len(atributos[0])
map_size = 7
sigma = 7
longer_width = 0

if red_desde_archivo:
    SOM = encoder.from_json(red_desde_archivo, 2)
else:
    SOM = som.SelfOrganizedMap(n_entrada, map_size, offset_width_map=longer_width)

# Si proveo red del ej1 para redurcir, entonces no tomar los documentos enteros sino
# Reducir dimensionalidad utilizando dicha red
if red_ej1 is not None:
    coordenadas = SOM.translate_documentos_to_coordenadas(dataset_train, red_ej1)
    SOM.train_con_documentos(coordenadas, sigma=sigma, epochs=epochs)
    resultados = SOM.predict(coordenadas, categorias_verificacion)

# Si no hay red ej1, usar dataset default
else:
    SOM.train_con_documentos(dataset_train, sigma=sigma, epochs=epochs)
    resultados = SOM.predict(dataset_train, categorias_verificacion)


########## OUTPUT A JSON ##############
if red_hacia_archivo:
    encoder.to_json(red_hacia_archivo, SOM, 2)

########## OBTENCION COORDENADAS ##############

cmap = colors.ListedColormap(
    [
        'white', 
        (51.0/256.0, 51.0/256.0, 51.0/256.0, 1), 
        (33.0/256.0, 150.0/256.0, 243.0/256.0, 1),
        (76.0/256.0, 175.0/256.0, 80.0/256.0, 1),
        (244.0/256.0, 67.0/256.0, 54.0/256.0, 1),
        (255.0/256.0, 235.0/256.0, 59.0/256.0, 1),
        (121.0/256.0, 85.0/256.0, 72.0/256.0, 1),
        (255.0/256.0, 152.0/256.0, 0.0/256.0, 1),
        (156.0/256.0, 39.0/256.0, 176.0/256.0, 1),
        (96.0/256.0, 125.0/256.0, 139.0/256.0, 1),
        (63.0/256.0, 81.0/256.0, 181.0/256.0, 1)
    ]
)
bounds = range(11)
norm = colors.BoundaryNorm(bounds, cmap.N)

column_labels = range(map_size)
row_labels = range(map_size+longer_width)
heatmap = plt.pcolor(np.array(resultados), cmap=cmap, norm=norm)
heatmap.axes.set_xticklabels = column_labels
heatmap.axes.set_yticklabels = row_labels
m = plt.colorbar(heatmap, ticks=range(11))

# plt.show()

######### CALCULO DEL ERROR CON VALIDACION ###########

if filepath == "sanger_network.data":
    resultados_validation = SOM.predict(coordenadas, categorias_verificacion, len(matrix) * 0.9)
else:
    resultados_validation = SOM.predict(dataset_validation, categorias_verificacion, int(len(matrix) * 0.9))

errores_x_categoria = [(x+1, 0) for x in range(9)]

for i, row in enumerate(resultados_validation):
    for j, _ in enumerate(row):
        if resultados[i][j] != resultados_validation[i][j]:
            errores_x_categoria[resultados[i][j] - 1] = (errores_x_categoria[resultados[i][j] - 1][0], errores_x_categoria[resultados[i][j] - 1][1] + 1)

fig = plt.figure()
chart_bar = fig.add_subplot(111)

y = [q[1] for q in errores_x_categoria]
x = [q[0] for q in errores_x_categoria]
width = 1/1.5
chart_bar.bar(x, y, width, color=(63.0/256.0, 81.0/256.0, 181.0/256.0, 1))


# print resultados
# print '---------------'
# print resultados_validation
# print errores_x_categoria

plt.show()

