# -*- coding: utf-8 -*-
import argparse
import os

def iniciar():
    usage = 'Este script tiene un único parametro obligatorio, que es el numero de ejercicio del TP. Puede ser 1 o 2 \n' \
          'Todos los demás son opcionales.\n' \
          'Ejemplo de ejecución: \n' \
          '$ python script.py 1 -ep=10000 -eta=0.01 -tr=20 -te=30 -val=50 -fa=tangente -dp=normal -tambatch=1 -mo=0'

    parser = argparse.ArgumentParser(usage=usage)

    # Argumento obligatorio: ejercicio a resolver
    parser.add_argument("nro_ejercicio", type=str, help='Numero de ejercicio. Valores: 1/2')

    # Argumentos opcionales:

    parser.add_argument("-file", "--filepath", default='tp2_training_dataset.csv', help='Ubicacion del archivo con los datasets a procesar')
    parser.add_argument("-ep", "--epochs", default=200, help='Cantidad de epocas. Default = 500')
    parser.add_argument("-eta", "--eta", default=0.01, help='Tasa de aprendizaje. Default = 0.05')

    parser.add_argument("-tr", "--train", default=90, help='% de input a utilizar como training. Default = 70')
    parser.add_argument("-te", "--test", default=0, help='% de input a utilizar como testing. Default = 20')
    parser.add_argument("-val", "--validation", default=10, help='% de input a utilizar como validation. Default = 10')

    parser.add_argument("-rda", "--red_desde_archivo", default=None,
                        help='Permite elegir una red ya entrenada. Las redes estan almacenadas en archivos.'
                             'Este parametro toma un filepath que contenga un txt con una red. Opciones: red_ej1.txt, red_ej2.txt')

    parser.add_argument("-rha", "--red_hacia_archivo", default=None,
                        help='Permite elegir una red ya entrenada. Las redes estan almacenadas en archivos json.'
                             'Este parametro toma un filepath que contenga una red en formato json. Opciones: red_ej1.json, red_ej2.json')

    parser.add_argument("-r", "--regla", default="sanger", help='Regla de aprendizaje para ejercicio 1. Valores = oja/sanger')


    args = parser.parse_args()

    nro_ejercicio = args.nro_ejercicio
    filepath = args.filepath

    eta = float(args.eta)
    epochs = int(args.epochs)
    train_pct = float(args.train)
    test_pct = float(args.test)
    validation_pct = float(args.validation)
    regla = args.regla

    red_desde_archivo = args.red_desde_archivo
    red_hacia_archivo = args.red_hacia_archivo

    os.system('clear')
    print 'TP2 - Aprendizaje No Supervisado'
    print "Se intentará procesar los datos del ejercicio "+nro_ejercicio+" ejecutando "+str(epochs)+" épocas con ETA "+str(eta)
    print str(train_pct) + "% del input utilizado como Entrenamiento"
    print str(test_pct) + "% del input utilizado como Testing"
    print str(validation_pct) + "% del input utilizado como Validacion"
    print "Regla de aprendizaje:" + regla
    print "Red a Utilizar: " + (red_desde_archivo if (red_desde_archivo is not None) else 'Nueva')
    print '-------------------------------------------------------------------------'

    return nro_ejercicio, filepath, eta, epochs, train_pct, test_pct, \
           validation_pct, regla, red_desde_archivo, red_hacia_archivo
