
import pandas as pd
#leemos el csv con los datos de cadda equipo
df_equipos = pd.read_csv('csvs/datos_fut.csv', encoding='utf-8', sep=';')

#creamos el dataframe 
columnas = ['fecha', 'local', 'visitante', 'prob_ganar_local', 'prob_empate', 'prob_ganar_visitante', 'resultado']
df_partidos = pd.DataFrame(columns=columnas)

#añadimos los datos de los grupos por días
df_partidos['fecha'] = ['13/02/24', '13/02/24', '14/02/24', '14/02/24', '20/02/24', '20/02/24', '21/02/24', '21/02/24', '5/03/24', '5/03/24', '6/03/24', '6/03/24', '12/03/24', '12/03/24', '13/03/24', '13/03/24']
df_partidos['local'] =['B 1903 København','RB Leipzig', 'SS Lazio','Paris Saint-Germain', 'FC Inter Turku' , 'PSV Eindhoven', 'FC Porto', 'SSC Napoli', 'FC Bayern München', 'Real Sociedad de Fútbol', 'Manchester City FC', 'Real Madrid CF', 'Arsenal FC', 'FC Barcelona', 'Club Atlético de Madrid', 'Borussia Dortmund']
df_partidos['visitante'] =['Manchester City FC', 'Real Madrid CF', 'FC Bayern München', 'Real Sociedad de Fútbol', 'Club Atlético de Madrid', 'Borussia Dortmund', 'Arsenal FC', 'FC Barcelona', 'SS Lazio', 'Paris Saint-Germain', 'B 1903 København', 'RB Leipzig', 'FC Porto', 'SSC Napoli', 'FC Inter Turku', 'PSV Eindhoven'] 


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
def actualizar_probabilidades(df_partidos):
    for i, fila in df_partidos.iterrows():
        prob_ganar_local, prob_empate, prob_gana_visitante = prob_enfrentada(fila['local'], fila['visitante'])
        df_partidos.at[i, 'prob_ganar_local'] = prob_ganar_local
        df_partidos.at[i, 'prob_empate'] = prob_empate
        df_partidos.at[i, 'prob_ganar_visitante'] = prob_gana_visitante
    return df_partidos
df_partidos = actualizar_probabilidades(df_partidos)


#lo pasamos a csv
df_partidos.to_csv('csvs/partidos_fut.csv', sep=';', index=False, encoding='utf-8')

