import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#leemos el csv
df = pd.read_csv('datos_fut.csv', encoding='utf-8', sep=';')
print(df.head())  

#vemos las columnas y los datos
print(df.columns)
print(df.info())  

    
