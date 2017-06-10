# -*- coding: utf-8 -*-
import numpy as np

class UnsupervisedLearningNetwork(object):

    def __init__(self, n_entrada, n_salida):
        self.pesos_red = list()

        for _ in range(n_salida):
            self.pesos_red.append({'pesos': np.random.uniform(-0.1, 0.1, n_entrada)})

    def train(self, dataset, eta=0.05, epochs=100):

        for _ in range(epochs):
            
            for _, fila in enumerate(dataset):

                for n_neurona in range(len(self.pesos_red)):

                    salida_neurona = self.neuron_output(n_neurona, fila)
                    self.pesos_red[n_neurona]['pesos'] = self.weight_update_hebbian(n_neurona, salida_neurona, eta)
                    print salida_neurona
        
        return self

    def neuron_output(self, neurona, entrada_neurona):
        return np.dot(entrada_neurona, self.pesos_red[neurona]['pesos'])

    def weight_update_hebbian(self, neurona, salida_neurona, eta):
        updated = list()

        for _, peso in enumerate(self.pesos_red[neurona]['pesos']):
            updated.append(eta * peso * salida_neurona) 

        return updated
