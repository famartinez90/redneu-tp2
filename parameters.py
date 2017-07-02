# -*- coding: utf-8 -*-
import argparse
import os

def iniciar():
    parser = argparse.ArgumentParser()

    parser.add_argument("-file", "--filepath", default='tp2_training_dataset.csv', help='Ubicacion del archivo con los datasets a procesar')
    parser.add_argument("-ep", "--epochs", default=200, help='Cantidad de epocas. Default = 500')
    parser.add_argument("-eta", "--eta", default=0.01, help='Tasa de aprendizaje. Default = 0.05')

    parser.add_argument("-rda", "--red_desde_archivo", default=None,
                        help='Permite elegir una red ya entrenada. Las redes estan almacenadas en archivos.'
                             'Este parametro toma un filepath que contenga un txt con una red. Opciones: red_ej1.txt, red_ej2.txt')

    parser.add_argument("-rha", "--red_hacia_archivo", default=None,
                        help='Permite elegir una red ya entrenada. Las redes estan almacenadas en archivos json.'
                             'Este parametro toma un filepath que contenga una red en formato json. Opciones: red_ej1.json, red_ej2.json')

    parser.add_argument("-r", "--regla", default="sanger", help='Regla de aprendizaje para ejercicio 1. Valores = oja/sanger')
    parser.add_argument("-dim", "--dimensiones", default=3,
                        help='Dimension de salida para ejercicio 1 y de entrada para el ejercicio 2')

    parser.add_argument("-red_ej1", "--red_ej1", default=None,
                        help='Red del ejercicio 1, para utilizarla en el ejercicio 2 para reducir dimensiones.')


    args = parser.parse_args()

    filepath = args.filepath

    eta = float(args.eta)
    epochs = int(args.epochs)
    regla = args.regla
    dimensiones = int(args.dimensiones)

    red_desde_archivo = args.red_desde_archivo
    red_hacia_archivo = args.red_hacia_archivo

    red_ej1 = args.red_ej1

    os.system('clear')
    print 'TP2 - Aprendizaje No Supervisado'
    print "Se intentará procesar los datos del ejercicio ejecutando "+str(epochs)+" épocas con ETA "+str(eta)
    print "Regla de aprendizaje:" + regla
    print "Red a Utilizar: " + (red_desde_archivo if (red_desde_archivo is not None) else 'Nueva')
    print '-------------------------------------------------------------------------'

    return filepath, eta, epochs, regla, dimensiones, red_desde_archivo, red_hacia_archivo, red_ej1

