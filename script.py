# -*- coding: utf-8 -*-
import csv
import network as ppn

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

######### TRAINING ##############

n_entrada = 850
n_salida = 3

PPN = ppn.UnsupervisedLearningNetwork(n_entrada, n_salida)
PPN.train(atributos, algoritmo="oja")
