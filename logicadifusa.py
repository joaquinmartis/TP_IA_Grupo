import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt

def logica_difusa(valor_x):
	x_in  = np.arange(-20, 21, 1)
	y_out = np.arange(-2.46, 16.0, 1.0)

	#Genero funciones de pertenencia difusas

	x_vec_baja=[-20,-15,-6,-3]
	x_f_pert_baja=fuzz.trapmf(x_in,x_vec_baja)
	x_vec_media=[-6,-3,3,6]
	x_f_pert_media=fuzz.trapmf(x_in,x_vec_media)
	x_vec_alta=[3,6,15,20]
	x_f_pert_alta=fuzz.trapmf(x_in,x_vec_alta)

	y_vec_baja= [-2.46, -1.46,1.46,2.46]
	y_f_pert_baja=fuzz.trapmf(y_out,y_vec_baja)
	y_vec_media= [1.46,2.46,5.0,7.0]
	y_f_pert_media=fuzz.trapmf(y_out,y_vec_media)
	y_vec_alta=[5.0,7.0,13.0,15.0]
	y_f_pert_alta=fuzz.trapmf(y_out,y_vec_alta)

	#fig, ax0 = plt.subplots(figsize=(10,3))
#
	#ax0.plot(x_in, x_f_pert_baja, 'b', linewidth=1.5, label='Mala')
	#ax0.plot(x_in, x_f_pert_media, 'g', linewidth=1.5, label='Normal')
	#ax0.plot(x_in, x_f_pert_alta, 'r', linewidth=1.5, label='Excelente')
	#ax0.set_title('Calidad de la comida')
	#ax0.legend()
#
	#plt.tight_layout()
	#plt.show()

	#fig, ax0 = plt.subplots(figsize=(10,3))

	#ax0.plot(y_out,y_f_pert_baja, 'b', linewidth=1.5, label='Mala')
	#ax0.plot(y_out,y_f_pert_media, 'g', linewidth=1.5, label='Normal')
	#ax0.plot(y_out,y_f_pert_alta, 'r', linewidth=1.5, label='Excelente')
	#ax0.set_title('Calidad de la comida')
	#ax0.legend()

	#plt.tight_layout()
	#plt.show()

	#print(x_in)
	#print(x_f_pert_baja)
	#print(valor_x)
	x_in_level_lo = fuzz.interp_membership(x_in, x_f_pert_baja, valor_x)
	x_in_level_md = fuzz.interp_membership(x_in, x_f_pert_media, valor_x)
	x_in_level_hi = fuzz.interp_membership(x_in, x_f_pert_alta, valor_x)

	y_out_activation_r1= np.fmin(x_in_level_lo, y_f_pert_baja) #Regla 1
	y_out_activation_r2=np.fmin(x_in_level_md, y_f_pert_media) #Regla 2
	y_out_activation_r3=np.fmin(x_in_level_hi, y_f_pert_alta)  #Regla 3

	#print(x_in_level_lo)
	#print(x_in_level_md)
	#print(x_in_level_hi)

	#fig, ax0 = plt.subplots(figsize=(10,3))

	#ax0.plot(y_out,y_out_activation_r1, 'b', linewidth=1.5, label='R1')
	#ax0.plot(y_out,y_out_activation_r2, 'g', linewidth=1.5, label='R2')
	#ax0.plot(y_out,y_out_activation_r3, 'r', linewidth=1.5, label='R3')
	#ax0.set_title('Calidad de la comida')
	#ax0.legend()
	#plt.tight_layout()
	#plt.show()

	y_out_activation_suma=np.fmax(y_out_activation_r1,np.fmax(y_out_activation_r2,y_out_activation_r3))

	print(y_out_activation_suma)

	fig, ax0 = plt.subplots(figsize=(10,3))

	ax0.plot(y_out,y_out_activation_suma, 'b', linewidth=1.5, label='OutPut')
	
	ax0.set_title('Calidad de la comida')
	
	ax0.legend()
	

	defuzz_centroid = fuzz.defuzz(y_out, y_out_activation_suma, 'bisector') 

	print(defuzz_centroid)

	plt.tight_layout()
	
	plt.show()

def main():
	logica_difusa(5)

main()