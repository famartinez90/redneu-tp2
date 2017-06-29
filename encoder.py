# -*- coding: utf-8 -*-
import network as ppn
import numpy as np
import io, json
from json import JSONEncoder

def to_json(filepath, red):
    with io.open(filepath, 'w', encoding='utf-8') as f:
        pesos = red.pesos_red

        for pesos_capa in pesos:
            pesos_capa['pesos'] = pesos_capa['pesos'].tolist()

        f.write(unicode(json.dumps(pesos, ensure_ascii=False)))

def from_json(filepath):
    with open(filepath, 'r') as content_file:
        content = content_file.read()
        pesos = json.loads(content)

        for pesos_capa in pesos:
            pesos_capa['pesos'] = np.array(pesos_capa['pesos'])

    return ppn.UnsupervisedLearningNetwork(basic_init_pesos=pesos)
