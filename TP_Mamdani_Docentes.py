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
    centro=5
    ancho=-2
    return fuzz.sigmf(nota_examen,centro,ancho)    

def f_pert_nota_examen_media_baja(nota_examen):
    media=20
    sigma=3
    return fuzz.gaussmf(nota_examen,media,sigma)


def f_pert_nota_examen_media(nota_examen):
    media=60
    sigma=3
    return fuzz.gaussmf(nota_examen,media,sigma)


def f_pert_nota_examen_media_alta(nota_examen):
    media=80
    sigma=3
    return fuzz.gaussmf(nota_examen,media,sigma)


def f_pert_nota_examen_alta(nota_examen):
    media=90
    sigma=3
    return fuzz.gaussmf(nota_examen,media,sigma)

def f_pert_nota_examen_muy_alta(nota_examen):
    centro=95
    ancho=2
    return fuzz.sigmf(nota_examen,centro,ancho)



def calcula_aumento_nota_difuso(nota_examen=7,x2_in="Bueno"):
    """
        x1: Entrada para la regla difusa relativa a la nota del examen del alumno. Rango: [0,100]
        x2: Entrada para la regla difusa relativa al concepto del alumno: Puede tomar valor "Regular", "Bueno", "Excelente"

        @return: La funcion retorna la nota final que le corresponde al alumno
    """
    
    x1_f_pert_regular=funcion_pertenencia_concepto_regular(x_in)
    x1_f_pert_bueno=funcion_pertenencia_concepto_bueno(x_in)
    x1_f_pert_excelente=funcion_pertenencia_concepto_excelente(x_in)


    #x2 variable nota_examen [0,100]
    x2=np.arange(0,100,1)
    x2_f_pert_baja=f_pert_nota_examen_baja(x2)
    
    x2_f_pert_media_baja=f_pert_nota_examen_media_baja(x2)

    x2_f_pert_media=f_pert_nota_examen_media(x2)

    x2_f_pert_media_alta=f_pert_nota_examen_media_alta(x2)

    x2_f_pert_alta=f_pert_nota_examen_alta(x2)

    x2_f_pert_muy_alta=f_pert_nota_examen_muy_alta(x2)







    nota_final=0
    return nota_final



def calcula_nota_final(nota_examen,concepto):
    nota_final= np.round(nota_examen/10+calcula_aumento_nota_difuso(nota_examen))

    #Si la nota se pasa de 10 entonces le queda un 10
    return min(10,nota_final)

def main():
    nota_examen=100
    concepto="Excelente"
    nota_final=calcula_nota_final(nota_examen,concepto):


    pass