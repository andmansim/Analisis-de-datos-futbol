import pandas as pd

'''
¿Cómo funciona el fútbol?
-Grupos: Los clubs de futbol se reparten en 8 grupos de 4 equipos cada uno. Cada equipo juega 6 partidos, 
3 en casa y 3 fuera. En función del resutado se reparten una serie de puntos (3 por victoria, 
1 por empate y 0 por derrota). Solo avanzan a la siguiente fase los dos primeros de cada grupo.
-Eliminatorias: Estas empiezan después de los grupos. Se compone de los octavos, cuartos, semifinales y final.
  -Octavos: Después de los grupos, los 16 equipos que han pasado se emparejan en 8 enfrentamientos.
  Enfrentamientos: son dos partidos, 1 en casa y otro fuera. Pasa a la siguiente ronda el equipo que haya 
  marcado más goles en total. Si hay un empate en goles, se decidirá bajo unas ciertas reglas. 
  Se da más importancia a los goles marcados fuera de casa y se suele seguir esto para dictar quien pasa.  
  
  
  -Cuartos: Los 8 equipos ganadores se emparejan en 4 enfrentamientos. Sigue el mismo reghlamento que los 
  octavos. 
  
  -Semifinal: los mismo pero con 4 equipos.
  
  -Final: Los dos equipos que han ganado en las semifinales se enfrentan en un partido.
'''
columnas = ['fecha', 'local', 'visitante', 'prob_ganar_local', 'porb_empate', 'prob_ganar_visitante']
#creamos el dataframe 
df_partidos = pd.DataFrame(columns=columnas)

#añadimos los datos de los grupos por días
df_partidos['fecha'] =['19/09/23', '19/09/23','19/09/23', '19/09/23', '19/09/23', '19/09/23','19/09/23', '19/09/23']
df_partidos['local'] =['milan', 'young boys', 'psg', 'shakhtar', 'manchester city', 'lazio', 'barcelona', 'feyenoord']
df_partidos['visitante'] =['newcastle', 'rb leipzig', 'dortmund', 'porto', 'estrella roja', 'altético madrid', 'antwerp', 'celtic fc']
print(df_partidos.head()) 