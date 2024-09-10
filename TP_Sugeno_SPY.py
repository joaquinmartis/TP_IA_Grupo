import csv 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
import time
from substractive_clustering import substractive_clustering
from fis import *
import datetime as dt

dates = ['01/02/1991','01/03/1991','01/04/1991']
x = [dt.datetime.strptime(d,'%m/%d/%Y').date() for d in dates]

def main():
	csvFile = pd.read_csv('spy.csv')
	print(csvFile)
	csvFile.to_numpy()
	csvTranspuesto=csvFile.T
	datos_close=csvFile["Close"]
	datos_date=csvFile["Date"]
	datos_y=datos_close.to_numpy()
	datos_x= [dt.datetime.strptime(date,'%d/%m/%y').date() for date in datos_date.to_numpy()]
	
	plt.plot(datos_x,datos_y)

	plt.show()



if __name__=="__main__":
	main()