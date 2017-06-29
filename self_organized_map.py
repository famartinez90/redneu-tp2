# -*- coding: utf-8 -*-
import numpy as np
from math import e, sqrt, pi, log

class SelfOrganizedMap(object):

    def __init__(self, n_entrada, map_size):
        matrix = np.array()

        for _ in range(map_size):
            row = np.array()
            
            for _ in range(map_size):
                row.append({'pesos': np.random.uniform(-0.1, 0.1, n_entrada)})

            matrix.append(row)

        self.map = matrix

    def train_con_documentos(self, documentos, categorias, sigma=1, eta=0.1, epochs=10):
        iteration = 0
        sigma_0 = sigma
        eta_0 = eta
        t1_sigma = 1000 / log(sigma_0)
        t2_eta = 1000

        for ep in range(epochs):
            
            for _, documento in enumerate(documentos):
                winner_index = np.array()
                min_distancia = float('inf')

                # Calculo todas las salidas de las neuronas de mi mapa
                for i, row in range(len(self.map)):
                    for j, _ in range(len(row)):
                        distancia = self.distancia_geometrica(documento, self.map[i][j]['pesos'])

                        # Voy calculando cual es la neurona ganadora
                        if distancia < min_distancia:
                            min_distancia = distancia
                            winner_index = np.array([i, j])
            
                for i, row in range(len(self.map)):
                    for j, _ in range(len(row)):

                        # Actualizacion de los pesos de las
                        # neuronas excitadas
                        h = self.funcion_vecindad(np.array([i, j]), winner_index, sigma)

                        self.map[i][j]['pesos'] += eta * h * np.subtract(documento, self.map[i][j]['pesos'])

            print 'Finalizada epoca '+str(ep)

        sigma, eta = self.cooling(iteration, sigma_0, t1_sigma, eta_0, t2_eta)

        return self

    def funcion_vecindad(self, neurona, ganadora, sigma):
        return e^(-(self.distancia_geometrica(neurona, ganadora)**2 / 2*(sigma**2)))

    def distancia_geometrica(self, rj, ri):
        return np.linalg.norm(rj-ri)

    def cooling(self, iteration_number, sigma_0, t1_sigma, eta_0, t2_eta):
        new_sigma = sigma_0 * (e ^ (-(iteration_number / t1_sigma)))
        new_eta = eta_0 * (e ^ (-(iteration_number / t2_eta)))

        if new_eta < 0.01:
            new_eta = 0.01

        return new_sigma, new_eta
        

