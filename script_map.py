# -*- coding: utf-8 -*-
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import self_organized_map as som
import parameters as params
import encoder as encoder


######### PARSEO DE PARAMETROS ##############

# nro_ejercicio, filepath, eta, epochs, train_pct, test_pct, \
#            validation_pct, regla, red_desde_archivo, red_hacia_archivo = params.iniciar()

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
map_size = 10
sigma = 5

SOM = som.SelfOrganizedMap(n_entrada, map_size)
SOM.train_con_documentos(dataset_train, categorias_verificacion, sigma=sigma)


# ######## OBTENCION COORDENADAS ##############

# coordenadas = []

# for documento in dataset_train:
#     coordenadas.append(PPN.predict_coordenadas_ej1(documento))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(
#     [vector[0] for vector in coordenadas], 
#     [vector[1] for vector in coordenadas], 
#     [vector[2] for vector in coordenadas], 
#     c=categorias_verificacion[:int(len(categorias_verificacion) * 0.9)]
# )

# plt.show()

# # Para datos de validacion

# coordenadas = []

# for documento in dataset_validation:
#     coordenadas.append(PPN.predict_coordenadas_ej1(documento))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(
#     [vector[0] for vector in coordenadas], 
#     [vector[1] for vector in coordenadas], 
#     [vector[2] for vector in coordenadas], 
#     c=categorias_verificacion[int(len(categorias_verificacion) * 0.9):]
# )

# plt.show()

# ######## OUTPUT A JSON ##############
# if red_hacia_archivo:
#     encoder.to_json(red_hacia_archivo, PPN)
