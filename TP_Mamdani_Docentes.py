import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
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
        


def f_pert_nota_examen_baja(nota_examen):
    centro=5
    ancho=-1
    return fuzz.sigmf(nota_examen,centro,ancho)    

def f_pert_nota_examen_media_baja(nota_examen):
    μ=25
    σ=9
    return fuzz.gaussmf(nota_examen,μ,σ)


def f_pert_nota_examen_media(nota_examen):
    μ=60
    σ=8
    return fuzz.gaussmf(nota_examen,μ,σ)


def f_pert_nota_examen_media_alta(nota_examen):
    μ=75
    σ=5
    return fuzz.gaussmf(nota_examen,μ,σ)


def f_pert_nota_examen_alta(nota_examen):
    μ=85
    σ=5
    return fuzz.gaussmf(nota_examen,μ,σ)

def f_pert_nota_examen_muy_alta(nota_examen):
    centro=95
    ancho=2
    return fuzz.sigmf(nota_examen,centro,ancho)

def f_pert_decremento_nota_final(incremento_nota):
    centro=-1
    ancho=0.3
    return fuzz.gaussmf(incremento_nota,centro,ancho)

def f_pert_neutralidad_nota_final(incremento_nota):
    centro=0
    ancho=0.3
    return fuzz.gaussmf(incremento_nota,centro,ancho)

def f_pert_incremento_nota_final(incremento_nota):
    centro=1
    ancho=0.3
    return fuzz.gaussmf(incremento_nota,centro,ancho)

def maximo_de_las_reglas(vec_rules):
    aux=vec_rules[0]
    for rule in vec_rules:
        aux=np.fmax(aux,rule)
    return aux

def calcula_aumento_nota_difuso(nota_examen=7,nota_concepto="Bueno"):
    """
        nota_examen: Entrada para la regla difusa relativa a la nota del examen del alumno. Rango: [0,100]
        nota_concepto: Entrada para la regla difusa relativa al concepto del alumno: Puede tomar valor "Regular", "Bueno", "Excelente"

        @return: La funcion retorna la nota final que le corresponde al alumno
    """
    
    nc_f_pert_regular=funcion_pertenencia_concepto_regular(nota_concepto)
    nc_f_pert_bueno=funcion_pertenencia_concepto_bueno(nota_concepto)
    nc_f_pert_excelente=funcion_pertenencia_concepto_excelente(nota_concepto)
    

    #ne variable nota_examen [0,100]
    ne=np.arange(0,101,1)
    ne_f_pert_baja=f_pert_nota_examen_baja(ne)
    ne_f_pert_media_baja=f_pert_nota_examen_media_baja(ne)
    ne_f_pert_media=f_pert_nota_examen_media(ne)
    ne_f_pert_media_alta=f_pert_nota_examen_media_alta(ne)
    ne_f_pert_alta=f_pert_nota_examen_alta(ne)
    ne_f_pert_muy_alta=f_pert_nota_examen_muy_alta(ne)
    
    fig, ax0 = plt.subplots(figsize=(8, 3))

    ax0.plot(ne, ne_f_pert_baja, 'b', linewidth=1.5, label='Baja')
    ax0.plot(ne, ne_f_pert_media_baja, 'g', linewidth=1.5, label='Media baja')
    ax0.plot(ne, ne_f_pert_media, 'r', linewidth=1.5, label='Media')
    ax0.plot(ne, ne_f_pert_media_alta, 'b', linewidth=1.5, label='Media alta')
    ax0.plot(ne, ne_f_pert_alta, 'g', linewidth=1.5, label='Alta')
    ax0.plot(ne, ne_f_pert_muy_alta, 'r', linewidth=1.5, label='Muy alta')
    ax0.legend()

    plt.tight_layout()

    plt.show()

    #y = salida de la regla difusa [-2, 2]
    y=np.arange(-2,2,0.1)
    y_negativo=f_pert_decremento_nota_final(y)
    y_neutro=f_pert_neutralidad_nota_final(y)
    y_positivo=f_pert_incremento_nota_final(y)

    fig, ax0 = plt.subplots(figsize=(8, 3))
    
    ax0.plot(y, y_negativo, 'b', linewidth=1.5, label='Negativo')
    ax0.plot(y, y_neutro, 'g', linewidth=1.5, label='Neutro')
    ax0.plot(y, y_positivo, 'r', linewidth=1.5, label='Positivo')
    # ax0.title('Salida de la regla difusa')
    ax0.legend()

    plt.tight_layout()

    plt.show()
    #Anda bien
    ne_in_baja = fuzz.interp_membership(ne, ne_f_pert_baja, nota_examen)
    ne_in_media_baja = fuzz.interp_membership(ne, ne_f_pert_media_baja, nota_examen)
    ne_in_media_baja=0.2
    ne_in_media = fuzz.interp_membership(ne, ne_f_pert_media, nota_examen)
    ne_in_media_alta = fuzz.interp_membership(ne, ne_f_pert_media_alta, nota_examen)
    ne_in_alta = fuzz.interp_membership(ne, ne_f_pert_alta, nota_examen)
    ne_in_muy_alta = fuzz.interp_membership(ne, ne_f_pert_muy_alta, nota_examen)

    vec_rules_negativo=[

        np.fmin(ne_in_media, nc_f_pert_regular),        #Nota media y concepto Regular -> nota negativo
        np.fmin(ne_in_media_alta, nc_f_pert_regular),   #Nota media-alta y concepto Regular -> nota negativo
        np.fmin(ne_in_alta, nc_f_pert_regular),         #Nota alta y concepto Regular -> nota negativo
        np.fmin(ne_in_muy_alta, nc_f_pert_regular),     #Nota muy alta y concepto Regular -> nota negativo
    ]

    vec_rules_neutral=[
        np.fmin(ne_in_baja, nc_f_pert_regular),         #Nota baja y concepto Regular -> nota neutral
        np.fmin(ne_in_media_baja, nc_f_pert_regular),   #Nota media-baja y concepto Regular -> nota neutral
        np.fmin(ne_in_media_baja, nc_f_pert_bueno),     #Nota media-baja y concepto Bueno -> nota neutral
        np.fmin(ne_in_media, nc_f_pert_bueno),          #Nota media y concepto Bueno -> nota neutral
        np.fmin(ne_in_media_alta, nc_f_pert_bueno),     #Nota media-alta y concepto Bueno -> nota neutral
        np.fmin(ne_in_media_alta, nc_f_pert_excelente), #Nota media-alta y concepto Excelente -> nota neutral
        np.fmin(ne_in_alta, nc_f_pert_bueno),           #Nota alta y concepto Bueno -> nota neutral
        np.fmin(ne_in_muy_alta, nc_f_pert_bueno),       #Nota muy alta y concepto Bueno -> nota neutral
    ]
    vec_rules_positivo=[
        np.fmin(ne_in_baja, nc_f_pert_bueno),           #Nota baja y concepto Bueno -> nota positiva 
        np.fmin(ne_in_baja, nc_f_pert_excelente),       #Nota baja y concepto Excelente -> nota positiva   
        np.fmin(ne_in_media_baja, nc_f_pert_excelente), #Nota media-baja y concepto Excelente -> nota positiva
        np.fmin(ne_in_media, nc_f_pert_excelente),      #Nota media y concepto Excelente -> nota positiva
        np.fmin(ne_in_alta, nc_f_pert_excelente),       #Nota alta y concepto Excelente -> nota positiva
        np.fmin(ne_in_muy_alta, nc_f_pert_excelente)    #Nota muy alta y concepto Excelente -> nota positiva
    ]

    activacion_y_negativo=[np.fmin(y_negativo,rule) for rule in vec_rules_negativo]
    activacion_y_neutro=[np.fmin(y_neutro,rule) for rule in vec_rules_neutral]
    activacion_y_positivo=[np.fmin(y_positivo,rule) for rule in vec_rules_positivo]

    agreggated_activation=activacion_y_negativo+activacion_y_neutro+activacion_y_positivo

    fig, ax0 = plt.subplots(figsize=(8, 3))
    
    ax0.plot(y, maximo_de_las_reglas(activacion_y_negativo), 'b', linewidth=1.5, label='Negativo')
    ax0.plot(y, maximo_de_las_reglas(activacion_y_neutro), 'g', linewidth=1.5, label='Neutro')
    ax0.plot(y, maximo_de_las_reglas(activacion_y_positivo), 'r', linewidth=1.5, label='Positivo')
    # ax0.title('Salida de la regla difusa')
    ax0.legend()

    plt.tight_layout()

    plt.show()
    resultado_de_agregacion=maximo_de_las_reglas(agreggated_activation)

    fig, ax0 = plt.subplots(figsize=(8, 3))
    
    ax0.plot(y, resultado_de_agregacion, 'b', linewidth=1.5, label='positivo')

    #ax0.title("Salida de la regla difusa")
    ax0.legend()

    plt.tight_layout()

    plt.show()

    incremento_nota=fuzz.defuzz(y, resultado_de_agregacion, 'centroid')

    return incremento_nota

def calcula_nota_final(nota_examen,concepto):
    nota_final= np.round(nota_examen/10+calcula_aumento_nota_difuso(nota_examen=nota_examen,nota_concepto=concepto), 2)

    #Si la nota se pasa de 10 entonces le queda un 10
    return min(10,nota_final)

def main():
    nota_examen=80
    concepto="Excelente"
    nota_final=calcula_nota_final(nota_examen,concepto)
    print(nota_final)

    pass

main()