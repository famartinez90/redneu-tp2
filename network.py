# -*- coding: utf-8 -*-
import numpy as np

class UnsupervisedLearningNetwork(object):

    def __init__(self, n_entrada, n_salida):
        self.pesos_red = list()

        for _ in range(n_salida):
            self.pesos_red.append({'pesos': np.random.uniform(-0.1, 0.1, n_entrada)})

    def train_ej1(self, dataset, eta=0.01, epochs=10, algoritmo="sanger"):

        # for X en D:
        #     Y = X . W
        #     for j en [1..M]:
        #         for i en [1..N]:
        #             X~_i = 0
        #             for k en [1..Q]
        #                 X~_i += Y_k . W_ik
        #             DeltaW_ij = eta . (X_i - X~_i) . Y_j
        #     W += DeltaW

        for ep in range(epochs):
            
            for _, documento in enumerate(dataset):
                y = list()

                # Recorro las neuronas de salida 2 veces porque necesito tener
                # calculadas las salidas de las mismas para hacer Oja
                for n_neurona in range(len(self.pesos_red)):
                    salida_neurona = np.dot(documento, self.pesos_red[n_neurona]['pesos'])
                    y.append(salida_neurona)

                # Para todas las neuronas de salida, se actualizan los pesos de
                # las 850 entradas para cada documento
                for n_neurona in range(len(self.pesos_red)):           
                    delta_w = list()
                    
                    for i, atributo in enumerate(documento):
                        x = 0

                        # Las salidas utilizadas para actualizar los pesos dependen
                        # de si se usa Oja o Sanger
                        for k in range(self.calcular_intervalo(algoritmo, n_neurona)):
                            x += y[k] * self.pesos_red[k]['pesos'][i]

                        delta_w.append(eta * (atributo - x) * salida_neurona)

                    # Actualizacion de los pesos
                    self.pesos_red[n_neurona]['pesos'] = np.sum([self.pesos_red[n_neurona]['pesos'], delta_w], axis=0)
            
            print 'Finalizada epoca '+str(ep)

        return self

    def predict_coordenadas_ej1(self, documento):
        coordenadas = list()

        for n_neurona in range(len(self.pesos_red)):
            salida_neurona = np.dot(documento, self.pesos_red[n_neurona]['pesos'])
            coordenadas.append(salida_neurona)

        return coordenadas

    def calcular_intervalo(self, algoritmo, neurona_actual):
        if algoritmo == "hebb":
            return 0
        
        if algoritmo == "oja1":
            return 1

        if algoritmo == "oja":
            return len(self.pesos_red)
        
        if algoritmo == "sanger":
            return neurona_actual+1
