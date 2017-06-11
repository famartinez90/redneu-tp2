# -*- coding: utf-8 -*-
import network as ppn
import matplotlib.pyplot as plt

######### INICIO SCRIPT ##############

N_ENTRADA = 3
N_SALIDA = 1

PPN = ppn.UnsupervisedLearningNetwork(N_ENTRADA, N_SALIDA)
PPN.train([[0.5, 0.3, 0.7]], algoritmo="hebb")
