
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
import pandas as pd
#leemos el csv con los datos de cadda equipo
df_equipos = pd.read_csv('datos_fut.csv', encoding='utf-8', sep=';')

#creamos el dataframe 
columnas = ['fecha', 'local', 'visitante', 'prob_ganar_local', 'porb_empate', 'prob_ganar_visitante']
df_partidos = pd.DataFrame(columns=columnas)

#añadimos los datos de los grupos por días
df_partidos['fecha'] =['19/09/23', '19/09/23','19/09/23', '19/09/23', '19/09/23', '19/09/23','19/09/23', '19/09/23']
df_partidos['local'] =['AC Milan', 'BSC Young Boys', 'Paris Saint-Germain', 'FC Shakhtar Donetsk', 'Manchester City FC', 'SS Lazio', 'FC Barcelona', 'Feyenoord']
df_partidos['visitante'] =['Newcastle United FC', 'RB Leipzig', 'Borussia Dortmund', 'FC Porto', 'FK Crvena zvezda', 'Club Atlético de Madrid', 'Royal Antwerp FC', 'Celtic FC']
print(df_partidos.head()) 

def prob_enfrentada(equipo1, equipo2):

    #Recogemos los datos de cada equipo
    prob_gana_local = df_equipos[df_equipos['Club']== equipo1]['porganarpartido'].values[0]
    prob_gana_visitante = df_equipos[df_equipos['Club'] == equipo2]['porganarpartido'].values[0]
    prob_empate_local = df_equipos[df_equipos['Club'] == equipo1]['poremppartido'].values[0]
    prob_empate_visitante = df_equipos[df_equipos['Club'] == equipo2]['poremppartido'].values[0]
    prob_perder_local = df_equipos[df_equipos['Club'] == equipo1]['porperderpartido'].values[0]
    prob_perder_visitante = df_equipos[df_equipos['Club'] == equipo2]['porperderpartido'].values[0]
    
    #Calculamos las probabilidades enfrentadas
    prob_gana_local1 = round((prob_gana_local * prob_perder_visitante)/100, 2)
    prob_gana_visitante1 = round((prob_gana_visitante * prob_perder_local)/100, 2)
    prob_empate1 = round((prob_empate_local * prob_empate_visitante)/100, 2)

    return prob_gana_local1, prob_empate1, prob_gana_visitante1

#Añadimos las probabilidades enfrentadas al dataframe
for i, fila in df_partidos.iterrows():
    prob_ganar_local, prob_empate, prob_gana_visitante = prob_enfrentada(fila['local'], fila['visitante'])
    df_partidos.at[i, 'prob_ganar_local'] = prob_ganar_local
    df_partidos.at[i, 'porb_empate'] = prob_empate
    df_partidos.at[i, 'prob_ganar_visitante'] = prob_gana_visitante
print(df_partidos.head())

#lo pasamos a csv
df_partidos.to_csv('partidos_fut.csv', sep=';', index=False, encoding='utf-8')

