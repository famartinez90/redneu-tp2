# -*- coding: utf-8 -*-
import numpy as np
from math import e, sqrt, pi

class SelfOrganizedMap(object):

    def __init__(self, n_entrada, map_size):
        neuronas = np.array()
        
        for _ in range(map_size):
            neuronas.append({'valor': 0, 'pesos': np.random.uniform(-0.1, 0.1, n_entrada)})

        self.map = np.matrix(neuronas)

    def train(self, dataset, eta=0.01, epochs=10):
        # TODO
        return self

    def funcion_vecindad(self):
        # TODO
        return 0

    def gauss(self, x, iteration_number):
        media = 0
        ancho = 2

        return 1/(sqrt(2*pi)*ancho)*e**(-0.5*(float(x-media)/ancho)**2)

    def cooling(self, area_parameter, eta, iteration_number):
        # TODO
        return 0
        

