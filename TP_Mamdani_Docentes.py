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
    centro=20
    ancho=-1
    return fuzz.sigmf(nota_examen,centro,ancho)    

def f_pert_nota_examen_media_baja(nota_examen):
    μ=30
    σ=8.5
    return fuzz.gaussmf(nota_examen,μ,σ)


def f_pert_nota_examen_media(nota_examen):
    μ=50
    σ=8
    return fuzz.gaussmf(nota_examen,μ,σ)


def f_pert_nota_examen_media_alta(nota_examen):
    μ=70
    σ=8
    return fuzz.gaussmf(nota_examen,μ,σ)


def f_pert_nota_examen_alta(nota_examen):
    μ=85
    σ=5
    return fuzz.gaussmf(nota_examen,μ,σ)

def f_pert_nota_examen_muy_alta(nota_examen):
    centro=95
    ancho=1
    return fuzz.sigmf(nota_examen,centro,ancho)

def f_pert_decremento_nota_final(incremento_nota,impacto_concepto):
    centro=-impacto_concepto*0.5
    ancho=-10.0 / abs(impacto_concepto)
    return fuzz.sigmf(incremento_nota,centro,ancho)

def f_pert_decrementoModerado_nota_final(incremento_nota,impacto_concepto):
    inicio= - np.round(impacto_concepto*0.5)
    pico=-np.round(impacto_concepto*0.25)
    fin= 0
    return fuzz.trimf(incremento_nota, [inicio, pico, fin])

def f_pert_incrementoModerado_nota_final(incremento_nota,impacto_concepto):
    inicio= 0
    pico= np.round(impacto_concepto*0.25)
    fin=np.round(impacto_concepto*0.5)
    return fuzz.trimf(incremento_nota, [inicio, pico, fin])

def f_pert_incremento_nota_final(incremento_nota,impacto_concepto):
    centro=impacto_concepto*0.5
    ancho= 10.0 / abs(impacto_concepto)
    return fuzz.sigmf(incremento_nota,centro,ancho)

def maximo_de_las_reglas(vec_rules):
    aux=vec_rules[0]
    for rule in vec_rules:
        aux=np.fmax(aux,rule)
    return aux

def calcula_aumento_nota_difuso(nota_examen,nota_concepto,impacto_concepto):
    """
        nota_examen: Entrada para la regla difusa relativa a la nota del examen del alumno. Rango: [0,100]
        nota_concepto: Entrada para la regla difusa relativa al concepto del alumno: Puede tomar valor "Regular", "Bueno", "Excelente"

        @return: La funcion retorna el cambio de nota que se aplica sobre la nota. Rango [0,impacto_concepto]
    """

    #Calculo de los valores de verdad de nota de concepto
    nc_f_pert_regular=funcion_pertenencia_concepto_regular(nota_concepto) #"p= nota de concepto es regular"
    nc_f_pert_bueno=funcion_pertenencia_concepto_bueno(nota_concepto) #"p = nota de concepto es bueno"
    nc_f_pert_excelente=funcion_pertenencia_concepto_excelente(nota_concepto) #"p = nota de concepto es excelente"
    
    #Se define el conjunto soporte para la nota de examen
    x_nota=np.arange(0,101,1)

    #Define las funciones de pertenencia de las notas de examen: baja ,media baja ,media, media alta, alta y muy alta.
    ne_f_pert_baja=f_pert_nota_examen_baja(x_nota)
    ne_f_pert_media_baja=f_pert_nota_examen_media_baja(x_nota)
    ne_f_pert_media=f_pert_nota_examen_media(x_nota)
    ne_f_pert_media_alta=f_pert_nota_examen_media_alta(x_nota)
    ne_f_pert_alta=f_pert_nota_examen_alta(x_nota)
    ne_f_pert_muy_alta=f_pert_nota_examen_muy_alta(x_nota)
    

    #Se muestran las funciones de pertenencia asociadas a las notas: baja ,media baja ,media, media alta, alta y muy alta.
    fig, ax1 = plt.subplots(figsize=(8, 3))
    ax1.plot(x_nota, ne_f_pert_baja, 'b', linewidth=1.5, label='Baja')
    ax1.plot(x_nota, ne_f_pert_media_baja, 'g', linewidth=1.5, label='Media baja')
    ax1.plot(x_nota, ne_f_pert_media, 'r', linewidth=1.5, label='Media')
    ax1.plot(x_nota, ne_f_pert_media_alta, 'c', linewidth=1.5, label='Media alta')
    ax1.plot(x_nota, ne_f_pert_alta, 'm', linewidth=1.5, label='Alta')
    ax1.plot(x_nota, ne_f_pert_muy_alta, 'y', linewidth=1.5, label='Muy alta')
    ax1.set_title("Nota de examen")
    ax1.legend()
    plt.tight_layout()
    plt.show(block=False)

    #Se define el conjunto soporte para la nota de examen
    y=np.arange(-impacto_concepto,impacto_concepto+1,1)

    #Se establecen las funciones de pertenencia para la salida
    y_negativo=f_pert_decremento_nota_final(y,impacto_concepto)
    y_negativomoderado=f_pert_decrementoModerado_nota_final(y,impacto_concepto)
    y_positivomoderado=f_pert_incrementoModerado_nota_final(y,impacto_concepto)
    y_positivo=f_pert_incremento_nota_final(y,impacto_concepto)

    #Se muestran las funciones de pertenencia asociadas a los cambios de nota: negativo, negativo moderado, positivo moderado o considerado
    fig, ax2 = plt.subplots(figsize=(8, 3))    
    ax2.plot(y, y_negativo, 'b', linewidth=1.5, label='Negativo')
    ax2.plot(y,  y_negativomoderado, 'g', linewidth=1.5, label='Negativo moderado')
    ax2.plot(y,  y_positivomoderado, 'y', linewidth=1.5, label='Positivo moderado')
    ax2.plot(y, y_positivo, 'r', linewidth=1.5, label='Considerado')
    ax2.legend()
    ax2.set_title("Cambio de nota")
    plt.tight_layout()
    plt.show(block=False)

    #Calcula el valor de verdad de la nota de examen: baja ,media baja ,media, media alta, alta y muy alta
    ne_in_baja = fuzz.interp_membership(x_nota, ne_f_pert_baja, nota_examen)
    ne_in_media_baja = fuzz.interp_membership(x_nota, ne_f_pert_media_baja, nota_examen)
    ne_in_media = fuzz.interp_membership(x_nota, ne_f_pert_media, nota_examen)
    ne_in_media_alta = fuzz.interp_membership(x_nota, ne_f_pert_media_alta, nota_examen)
    ne_in_alta = fuzz.interp_membership(x_nota, ne_f_pert_alta, nota_examen)
    ne_in_muy_alta = fuzz.interp_membership(x_nota, ne_f_pert_muy_alta, nota_examen)


    #Definicion de reglas y fuzzificacion

    """
    Metodología del profesor:
        -Con las notas bajas y media bajas se es muy exigente: el concepto negativo resta mucho y el concepto bien o excelente suman muy poco a la nota
        -Las notas medias se consideran muy sensibles, ya que segun esta nota el alumno puede aprobar o desaprobar: el concepto regular descuenta moderadamente, el concepto bien aumenta moderadamente y el concepto excelente suma mucho
        -El profesor considera que las notas medias altas y altas son poco relevantes en el cambio de nota, pero de ellas se espera que el concepto sea bueno. Por lo tanto un concepto regular resta mucho y el concepto bien y excelente suma poco
        -Por ultimo, las notas muy altas son muy estimadas por el profesor, por lo tanto el concepto resta muy poco y el concepto bien y excelente suman mucho.
    """
    
    vec_rules_negativo=[
        np.fmin(ne_in_baja, nc_f_pert_regular),         #Nota baja y concepto Regular -> cambio negativo

        np.fmin(ne_in_media_baja, nc_f_pert_regular),   #Nota media-baja y concepto Regular -> cambio negativo

        np.fmin(ne_in_media_alta, nc_f_pert_regular),   #Nota media-alta y concepto Regular -> cambio negativo

        np.fmin(ne_in_alta, nc_f_pert_regular),         #Nota alta y concepto Regular -> cambio negativo
    ]

    vec_rules_negativo_moderado=[
        np.fmin(ne_in_media, nc_f_pert_regular),        #Nota media y concepto Regular -> cambio -moderado
        np.fmin(ne_in_muy_alta, nc_f_pert_regular),     #Nota muy alta y concepto Regular -> cambio -moderado
    ]

    vec_rules_positivo_moderado=[
        np.fmin(ne_in_baja, nc_f_pert_excelente),       #Nota baja y concepto Excelente -> cambio +moderado  
        np.fmin(ne_in_baja, nc_f_pert_bueno),           #Nota baja y concepto Bueno -> cambio +moderado

        np.fmin(ne_in_media_baja, nc_f_pert_excelente), #Nota media-baja y concepto Excelente -> cambio +moderado
        np.fmin(ne_in_media_baja, nc_f_pert_bueno),     #Nota media-baja y concepto Bueno -> cambio +moderado

        np.fmin(ne_in_media, nc_f_pert_bueno),          #Nota media y concepto Bueno -> cambio +moderado

        np.fmin(ne_in_media_alta, nc_f_pert_bueno),     #Nota media-alta y concepto Bueno -> cambio +moderado
        np.fmin(ne_in_media_alta, nc_f_pert_excelente), #Nota media-alta y concepto Excelente -> cambio +moderado

        np.fmin(ne_in_alta, nc_f_pert_bueno),           #Nota alta y concepto Bueno -> cambio +moderado
        np.fmin(ne_in_alta, nc_f_pert_excelente),       #Nota alta y concepto Excelente -> cambio +moderado
    ]
    
    vec_rules_positivo=[
        np.fmin(ne_in_media, nc_f_pert_excelente),      #Nota media y concepto Excelente -> cambio considerado


        np.fmin(ne_in_muy_alta, nc_f_pert_bueno),       #Nota muy alta y concepto Bueno -> cambio considerado
        np.fmin(ne_in_muy_alta, nc_f_pert_excelente)    #Nota muy alta y concepto Excelente -> cambio considerado
    ]


    #Inferencia
    activacion_y_negativo=[np.fmin(y_negativo,rule) for rule in vec_rules_negativo]
    activacion_y_negmoderado=[np.fmin(y_negativomoderado,rule) for rule in vec_rules_negativo_moderado]
    activacion_y_posmoderado=[np.fmin(y_positivomoderado,rule) for rule in vec_rules_positivo_moderado]
    activacion_y_positivo=[np.fmin(y_positivo,rule) for rule in vec_rules_positivo]
    
    fig, ax3 = plt.subplots(figsize=(8, 3))
    
    y0=np.zeros_like(y)

    mf_actneg=maximo_de_las_reglas(activacion_y_negativo)
    mf_actnegmod=maximo_de_las_reglas(activacion_y_negmoderado)
    mf_actposmod= maximo_de_las_reglas(activacion_y_posmoderado)
    mf_actpos= maximo_de_las_reglas(activacion_y_positivo)

    ax3.plot(y, mf_actneg, 'b', linewidth=1.5, label='Negativo')
    ax3.fill_between(y, y0, mf_actneg, facecolor='b', alpha=0.7)

    ax3.plot(y,mf_actnegmod , 'g', linewidth=1.5, label='Negativo Moderado')
    ax3.fill_between(y, y0, mf_actnegmod, facecolor='g', alpha=0.7)
    
    ax3.plot(y, mf_actposmod, 'y', linewidth=1.5, label='Positivo Moderado')
    ax3.fill_between(y, y0, mf_actposmod, facecolor='y', alpha=0.7) 

    ax3.plot(y, mf_actpos, 'r', linewidth=1.5, label='Positivo')
    ax3.fill_between(y, y0, mf_actpos, facecolor='r', alpha=0.7)

    ax3.set_title("Inferencia y agregacion parcial")
    ax3.set_ylim([0, 1+0.1])
    ax3.legend()
    plt.tight_layout()
    plt.show(block=False)

    resultado_de_inferencia=activacion_y_negativo+activacion_y_negmoderado+activacion_y_posmoderado+activacion_y_positivo
    resultado_de_agregacion= maximo_de_las_reglas (resultado_de_inferencia)
    incremento_nota=fuzz.defuzz(y, resultado_de_agregacion, 'centroid')

    fig, ax4 = plt.subplots(figsize=(8, 3))
    ax4.plot(y,  resultado_de_agregacion, 'g', linewidth=1.5)
    ax4.fill_between(y, y0, resultado_de_agregacion, facecolor='g', alpha=0.7)
    ax4.plot([incremento_nota, incremento_nota], [0, fuzz.interp_membership(y, resultado_de_agregacion, incremento_nota)], 'k', linewidth=1.5, alpha=0.9)
    ax4.set_ylim([0, 1+0.1])
    ax4.legend()
    ax4.set_title("Agregacion y desfuzzificacion")
    plt.show(block=False)

    
    return np.round(incremento_nota)

def calcula_nota_final(nota_examen,concepto,impacto_concepto):
    nota_final= int(np.trunc(nota_examen+calcula_aumento_nota_difuso(nota_examen,concepto,impacto_concepto)))

    #Si la nota se pasa de 10 entonces le queda un 10 o si la nota es menor a 0, le queda un 0
    return max(0,min(100,nota_final))

def main():
    
    entrada = input("Ingrese la nota de examen (0-100): ")
    while not( str.isdigit(entrada) and  0 <= int(entrada) <= 100 ):
        entrada = input("Error: ingrese correctamente la nota de examen (0,100): ")
    nota_examen= int(entrada)

    entrada = input("Ingrese la nota de concepto (regular/bueno/excelente): ").capitalize()
    while not( entrada in ["Excelente", "Regular", "Bueno"] ):
        entrada = input("Error: ingrese correctamente la nota de concepto (regular/bueno/excelente): ").capitalize()
    concepto= entrada

    entrada = input("Ingrese el máximo impacto en el concepto (4,20): ").capitalize()
    while not( str.isdigit(entrada) and  4<= int(entrada) <= 20  ):
        entrada = input("Error: ingrese correctamente el máximo impacto en el concepto (4-20):: ").capitalize()
    impacto_concepto= int(entrada)

    nota_final=calcula_nota_final(nota_examen,concepto,impacto_concepto)
    print(nota_final)
    input("Presione enter para finalizar")

main()