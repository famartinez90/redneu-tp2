# -*- coding: utf-8 -*-
import numpy as np

class UnsupervisedLearningNetwork(object):

    def __init__(self, n_entrada, n_salida):
        self.pesos_red = list()

        for _ in range(n_salida):
            self.pesos_red.append({'pesos': np.random.uniform(-0.1, 0.1, n_entrada)})

    def train(self, dataset, eta=0.05, epochs=100, algoritmo="hebb"):

        # for X en D:
        #     Y = X . W
        #     for j en [1..M]:
        #         for i en [1..N]:
        #             X~_i = 0
        #             for k en [1..Q]
        #                 X~_i += Y_k . W_ik
        #             DeltaW_ij = eta . (X_i - X~_i) . Y_j
        #     W += DeltaW

        for _ in range(epochs):
            
            for _, fila in enumerate(dataset):
                y = list()

                for n_neurona in range(len(self.pesos_red)):
                    salida_neurona = np.dot(fila, self.pesos_red[n_neurona]['pesos'])
                    y.append(salida_neurona)
                    delta_w = list()
                    
                    for i, entrada in enumerate(fila):
                        x = 0

                        for k in range(self.calcular_intervalo(algoritmo, n_neurona)):
                            x += y[k] * self.pesos_red[k]['pesos'][i]

                        delta_w.append(eta * (entrada - x) * salida_neurona)

                    self.pesos_red[n_neurona]['pesos'] = np.sum([self.pesos_red[n_neurona]['pesos'], delta_w], axis=0)
                    print salida_neurona
        
        return self

    def calcular_intervalo(self, algoritmo, neurona_actual):
        if algoritmo == "hebb":
            return 0
        
        if algoritmo == "oja1":
            return 1

        if algoritmo == "oja":
            return len(self.pesos_red)
        
        if algoritmo == "sanger":
            return neurona_actual+1
