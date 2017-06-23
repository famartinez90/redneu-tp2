# -*- coding: utf-8 -*-
import csv
import network as ppn
import numpy as np
import random

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

random.shuffle(matrix)

dataset_train = matrix[:int(len(matrix) * 0.9)]
dataset_validation = matrix[int(len(matrix) * 0.9):]

######## TRAINING ##############

n_entrada = len(atributos[0])
n_salida = 3

PPN = ppn.UnsupervisedLearningNetwork(n_entrada, n_salida)
PPN.train_ej1(dataset_train, algoritmo="oja", epochs=3)

######## OBTENCION COORDENADAS ##############

coordenadas = []

for documento in dataset_train:
    coordenadas.append(PPN.predict_coordenadas_ej1(documento))

print coordenadas
