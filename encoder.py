# -*- coding: utf-8 -*-
import network as ppn
import self_organized_map as som
import numpy as np
import io, json
from json import JSONEncoder

def to_json(filepath, red, numero_ejercicio):
    with io.open(filepath, 'w', encoding='utf-8') as f:

        if numero_ejercicio == 1:
            pesos = red.pesos_red

            for pesos_capa in pesos:
                pesos_capa['pesos'] = pesos_capa['pesos'].tolist()

            f.write(unicode(json.dumps(pesos, ensure_ascii=False)))
        else:
            pesos = red.map

            for i in pesos:
                for j in i:
                    pesos[i][j]['pesos'] = pesos[i][j]['pesos'].tolist()

            f.write(unicode(json.dumps(pesos, ensure_ascii=False)))

def from_json(filepath, numero_ejercicio):
    with open(filepath, 'r') as content_file:
        content = content_file.read()
        pesos = json.loads(content)

        if numero_ejercicio == 1:
            for pesos_capa in pesos:
                pesos_capa['pesos'] = np.array(pesos_capa['pesos'])

            return ppn.UnsupervisedLearningNetwork(basic_init_pesos=pesos)

        else:
            for i in pesos:
                for j in i:
                    pesos[i][j]['pesos'] = np.array(pesos[i][j]['pesos'])

            return som.SelfOrganizedMap(basic_init_pesos=pesos)