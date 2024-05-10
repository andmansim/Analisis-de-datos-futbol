
import pandas as pd
#leemos el csv con los datos de cadda equipo
df_equipos = pd.read_csv('csvs/datos_fut.csv', encoding='utf-8', sep=';')

#creamos el dataframe 
columnas = ['fecha', 'local', 'visitante','puntos_local', 'puntos_visitante', 'resultado']
df_partidos = pd.DataFrame(columns=columnas)

#añadimos los datos de los grupos por días
df_partidos['fecha'] = ['13/02/24', '13/02/24', '14/02/24', '14/02/24', '20/02/24', '20/02/24', '21/02/24', '21/02/24', '5/03/24', '5/03/24', '6/03/24', '6/03/24', '12/03/24', '12/03/24', '13/03/24', '13/03/24']
df_partidos['local'] =['B 1903 København','RB Leipzig', 'SS Lazio','Paris Saint-Germain', 'FC Inter Turku' , 'PSV Eindhoven', 'FC Porto', 'SSC Napoli', 'FC Bayern München', 'Real Sociedad de Fútbol', 'Manchester City FC', 'Real Madrid CF', 'Arsenal FC', 'FC Barcelona', 'Club Atlético de Madrid', 'Borussia Dortmund']
df_partidos['visitante'] =['Manchester City FC', 'Real Madrid CF', 'FC Bayern München', 'Real Sociedad de Fútbol', 'Club Atlético de Madrid', 'Borussia Dortmund', 'Arsenal FC', 'FC Barcelona', 'SS Lazio', 'Paris Saint-Germain', 'B 1903 København', 'RB Leipzig', 'FC Porto', 'SSC Napoli', 'FC Inter Turku', 'PSV Eindhoven'] 


def datos_nuevos(equipo1, equipo2):
    #Recogemos los datos de cada equipo
    porcapacidad_ofensiva_local = df_equipos[df_equipos['club']== equipo1]['porcapacidad_ofensiva'].values[0]
    porcapacidad_ofensiva_visitante = df_equipos[df_equipos['club'] == equipo2]['porcapacidad_ofensiva'].values[0]
    porcapacidad_defensiva_local = df_equipos[df_equipos['club'] == equipo1]['porcapacidad_defensiva'].values[0]
    porcapacidad_defensiva_visitante = df_equipos[df_equipos['club'] == equipo2]['porcapacidad_defensiva'].values[0]
    
    #comparamos los datosy les asignamos un número
    puntos_local = 0
    puntos_visitante = 0
    
    #Ofensiva A vs Ofensiva B 
    if porcapacidad_ofensiva_local > porcapacidad_ofensiva_visitante:
        puntos_local += 1
    else: 
        puntos_visitante += 1
    
    #Ofensiva A vs Defensiva B
    if porcapacidad_ofensiva_local > porcapacidad_defensiva_visitante:
        puntos_local += 1
    else: 
        puntos_visitante += 1
    
    #Defensiva A vs Ofensiva B
    if porcapacidad_defensiva_local > porcapacidad_ofensiva_visitante:
        puntos_local += 1
    else: 
        puntos_visitante += 1
        
   
    return puntos_local, puntos_visitante

def actualizar_datos(df_partidos):
    #añadimos los datos recogido anteriormente a un dataframe
    for i, fila in df_partidos.iterrows():
        puntos_local, puntos_visitante = datos_nuevos(fila['local'], fila['visitante'])
        df_partidos.at[i, 'puntos_local'] = puntos_local
        df_partidos.at[i, 'puntos_visitante'] = puntos_visitante
    return df_partidos

df_partidos = actualizar_datos(df_partidos)
#lo pasamos a csv
df_partidos.to_csv('csvs1/partidos_fut_regresion.csv', sep=';', index=False, encoding='utf-8')

