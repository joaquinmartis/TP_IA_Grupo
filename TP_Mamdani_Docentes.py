import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt


def funcion_pertenencia_concepto_regular(concepto):

    match concepto:
        case "Regular":
            return 1
        case "Bueno":
            return 0
        case "Excelente":
            return 0
        
def funcion_pertenencia_concepto_bueno(concepto):

    match concepto:
        case "Regular":
            return 0
        case "Bueno":
            return 1
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
        
def main():


    x1_in=0
    x2_in=0
    #x1 variable Concepto

    x1_f_pert_regular=4
    x1_f_pert_bueno=7
    x1_f_pert_excelente=10


    #x2 variable nota_examen [0,100]
    x2_vec_baja=[-20,-15,-6,-3]
    x2_f_pert_baja=fuzz.trapmf(x2_in,x2_vec_baja)
    x2_vec_media_baja=[-6,-3,3,6]
    x2_f_pert_media_baja=fuzz.trapmf(x2_in,x2_vec_media)
    x2_vec_media=[3,6,15,20]
    x2_f_pert_media=fuzz.trapmf(x2_in,x2_vec_alta)
    x2_vec_media_alta=[3,6,15,20]
    x2_f_pert_media_alta=fuzz.trapmf(x2_in,x2_vec_alta)
    x2_vec_alta=[3,6,15,20]
    x2_f_pert_alta=fuzz.trapmf(x2_in,x2_vec_alta)
    x2_vec_perfecto=[3,6,15,20]
    x2_f_pert_perfecto=fuzz.trapmf(x2_in,x2_vec_alta)





    pass