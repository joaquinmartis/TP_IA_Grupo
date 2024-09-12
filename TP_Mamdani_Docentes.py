import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt

def funcion_pertenencia_concepto_regular(concepto):

    match concepto:
        case "Regular":
            return 0.5
        case "Bueno":
            return 0
        case "Excelente":
            return 0
        
def funcion_pertenencia_concepto_bueno(concepto):

    match concepto:
        case "Regular":
            return 0
        case "Bueno":
            return 0.7
        case "Excelente":
            return 0
        
def funcion_pertenencia_concepto_excelente(concepto):

    match concepto:
        case "Regular":
            return 0
        case "Bueno":
            return 0
        case "Excelente":
            return 1
        
def f_pert_nota_examen_baja(nota_examen):
    x2_vec_baja=[0,0.2,1.8,2]
    return fuzz.trapmf(nota_examen,x2_vec_baja)

def f_pert_nota_examen_media_baja(nota_examen):
    x2_vec_media_baja=[0.8,2,3.8,4]
    return fuzz.trapmf(nota_examen,x2_vec_media_baja)


def f_pert_nota_examen_media(nota_examen):
    x2_vec_media=[3,6,15,20]
    return fuzz.trapmf(nota_examen,x2_vec_media)


def f_pert_nota_examen_media_alta(nota_examen):
    x2_vec_media_alta=[3,6,15,20]
    return fuzz.trapmf(nota_examen,x2_vec_media_alta)


def f_pert_nota_examen_alta(nota_examen):
    x2_vec_alta=[3,6,15,20]
    return fuzz.trapmf(nota_examen,x2_vec_alta)


def f_pert_nota_examen_muy_alta(nota_examen):
    x2_vec_muy_alta=[3,6,15,20]
    return fuzz.trapmf(nota_examen,x2_vec_muy_alta)


def calcula_nota_difuso(x1_in=7,x2_in="Bueno"):
    """
        x1: Entrada para la regla difusa relativa a la nota del examen del alumno. Rango: [0,100]
        x2: Entrada para la regla difusa relativa al concepto del alumno: Puede tomar valor "Regular", "Bueno", "Excelente"

        @return: La funcion retorna la nota final que le corresponde al alumno
    """

    x1_f_pert_regular=funcion_pertenencia_concepto_regular(x1_in)
    x1_f_pert_bueno=funcion_pertenencia_concepto_bueno(x1_in)
    x1_f_pert_excelente=funcion_pertenencia_concepto_excelente(x1_in)


    #x2 variable nota_examen [0,100]
    
    x2_f_pert_baja=
    
    x2_f_pert_media_baja=

    x2_f_pert_media=

    x2_f_pert_media_alta=

    x2_f_pert_alta=

    x2_f_pert_muy_alta=



    return nota_final





def main():



    pass