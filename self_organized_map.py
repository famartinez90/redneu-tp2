# -*- coding: utf-8 -*-
from math import e, log
import numpy as np
import encoder as encoder

class SelfOrganizedMap(object):

    def __init__(self, n_entrada = 1, map_size = 7, basic_init_pesos=None, offset_width_map=0):

        # Para cargar redes armadas con pesos ya entrenados
        if basic_init_pesos is not None:
            self.map = basic_init_pesos
        else:
            matrix = []

            for _ in range(map_size):
                row = []
                for _ in range(map_size+offset_width_map):
                    row.append({'pesos': np.random.uniform(-0.1, 0.1, n_entrada)})

                matrix.append(row)

            self.map = matrix

    def translate_documentos_to_coordenadas(self, documentos, sanger_network_file):
        ppn = encoder.from_json(sanger_network_file, 1)
        coordenadas = []

        for documento in documentos:
            coordenadas.append(ppn.predict_coordenadas_ej1(documento))

        return coordenadas

    def train_con_documentos(self, documentos, sigma=5, eta=0.1, epochs=10):
        iteration = 0.0
        sigma_0 = sigma
        eta_0 = eta
        t1_sigma = float(epochs) / float(log(sigma_0, 2))
        t2_eta = float(epochs)

        for ep in range(epochs):
            
            for _, documento in enumerate(documentos):
                winner_index = []
                min_distancia = float('inf')

                # Calculo todas las salidas de las neuronas de mi mapa
                for i, row in enumerate(self.map):
                    for j, _ in enumerate(row):
                        distancia = self.distancia_geometrica(documento, self.map[i][j]['pesos'])

                        # Voy calculando cual es la neurona ganadora
                        if distancia < min_distancia:
                            min_distancia = distancia
                            winner_index = np.array([i, j])
            
                for i, row in enumerate(self.map):
                    for j, _ in enumerate(row):

                        # Actualizacion de los pesos de las
                        # neuronas con la funcion de vecindad
                        h = self.funcion_vecindad(np.array([i, j]), winner_index, sigma)

                        self.map[i][j]['pesos'] += eta * h * np.subtract(documento, self.map[i][j]['pesos'])
            
            if iteration < (epochs * 3 / 4):
                sigma, eta = self.cooling(iteration, sigma_0, t1_sigma, eta_0, t2_eta)
            
            iteration += 1

            print 'Finalizada epoca '+str(ep)

        return self

    def predict(self, documentos, categorias):
        resultados = []

        for r1, row in enumerate(self.map):
            resultados.append([])

            for _ in enumerate(row):
                resultados[r1].append([])

        for k, documento in enumerate(documentos):
            winner_index = []
            min_distancia = float('inf')

            # Calculo todas las salidas de las neuronas de mi mapa
            for i, row in enumerate(self.map):
                for j, _ in enumerate(row):
                    distancia = self.distancia_geometrica(documento, self.map[i][j]['pesos'])

                    # Voy calculando cual es la neurona ganadora
                    if distancia < min_distancia:
                        min_distancia = distancia
                        winner_index = np.array([i, j])

            resultados[winner_index[0]][winner_index[1]].append(categorias[k])

        return self.determinar_categoria_ganadora(resultados)

    def validation_error(self, docs_validation, resultados_training, categorias, offset=0):
        error_x_categoria = [(x+1, 0) for x in range(9)]

        for k, documento in enumerate(docs_validation):
            winner_index = []
            min_distancia = float('inf')

            # Calculo todas las salidas de las neuronas de mi mapa
            for i, row in enumerate(self.map):
                for j, _ in enumerate(row):
                    distancia = self.distancia_geometrica(documento, self.map[i][j]['pesos'])

                    # Voy calculando cual es la neurona ganadora
                    if distancia < min_distancia:
                        min_distancia = distancia
                        winner_index = np.array([i, j])

            categoria_training = resultados_training[winner_index[0]][winner_index[1]]

            if categoria_training != categorias[k+offset]:
                print 'error en '+str(categoria_training)
                error_x_categoria[categoria_training] = (categoria_training, error_x_categoria[categoria_training][1]+1)

        return error_x_categoria

    def print_nice(self, data):
        for row in data:
            print row

    def determinar_categoria_ganadora(self, neurona_categorias):
        mas_comunes = []

        for i, neurona in enumerate(neurona_categorias):
            mas_comunes.append([])
            
            for _, categorias in enumerate(neurona):
                mas_comunes[i].append(self.mas_comun(categorias))

        return mas_comunes
                
    def mas_comun(self, categorias):
        if len(categorias) == 0:
            return 0

        return int(max(set(categorias), key=categorias.count))

    def funcion_vecindad(self, neurona, ganadora, sigma):
        return e**(-(self.distancia_geometrica(neurona, ganadora)**2 / 2*(sigma**2)))

    def distancia_geometrica(self, rj, ri):
        return float(np.linalg.norm(rj-ri))

    def cooling(self, iteration_number, sigma_0, t1_sigma, eta_0, t2_eta):
        new_sigma = sigma_0 * (e**(-(iteration_number / t1_sigma)))
        new_eta = eta_0 * (e**(-(iteration_number / t2_eta)))

        return new_sigma, new_eta
        

