import pandas as pd


def crear_datos(equipos_local, equipos_visitante, resultados):
    data = []
    for equipo_local, equipo_visitante, resultado in zip(equipos_local, equipos_visitante, resultados):
        datos_local = df_equipos[df_equipos['club'] == equipo_local].iloc[0]
        datos_visitante = df_equipos[df_equipos['club'] == equipo_visitante].iloc[0]
        data.append([equipo_local, equipo_visitante, datos_local['porganarpartido'], datos_visitante['porganarpartido'], datos_local['porperderpartido'], datos_visitante['porperderpartido'], datos_local['porcapacidad_ofensiva'], datos_visitante['porcapacidad_ofensiva'], datos_local['porcapacidad_defensiva'], datos_visitante['porcapacidad_defensiva'], resultado])
    return pd.DataFrame(data, columns=columnas)

def agregar_datos(df_existente, equipos_local_nuevos, equipos_visitante_nuevos, resultados_nuevos):
    nuevos_datos = []
    for equipo_local, equipo_visitante, resultado in zip(equipos_local_nuevos, equipos_visitante_nuevos, resultados_nuevos):
        datos_local = df_equipos[df_equipos['club'] == equipo_local].iloc[0]
        datos_visitante = df_equipos[df_equipos['club'] == equipo_visitante].iloc[0]
        nuevos_datos.append([equipo_local, equipo_visitante, datos_local['porganarpartido'], datos_visitante['porganarpartido'], datos_local['porperderpartido'], datos_visitante['porperderpartido'], datos_local['porcapacidad_ofensiva'], datos_visitante['porcapacidad_ofensiva'], datos_local['porcapacidad_defensiva'], datos_visitante['porcapacidad_defensiva'], resultado])
    df_nuevo = pd.DataFrame(nuevos_datos, columns=columnas)
    df_actualizado = pd.concat([df_existente, df_nuevo], ignore_index=True)
    return df_actualizado

#columnas
columnas = ['local', 'visitante', 'porganarpartido_local','porganarpartido_visitante','porperderpartido_local', 'porperderpartido_visitante', 'porcapacidad_ofensiva_local','porcapacidad_ofensiva_visitante', 'porcapacidad_defensiva_local','porcapacidad_defensiva_visitante', 'resultado']
#cogemos los datos de datos_fut.csv
df_equipos = pd.read_csv('csvs/datos_fut.csv', encoding='utf-8', sep=';')


#dividimos los equpos en dos listas, local y visitante 
#22-23
equipos_local = ['Eintracht Frankfurt', 'AC Milan','FC Internazionale Milano','Club Brugge','Liverpool FC', 'Borussia Dortmund', 'Paris Saint-Germain', 'RB Leipzig', 'SSC Napoli', 'FC Internazionale Milano', 'Real Madrid CF','FC Bayern München','AC Milan','Real Madrid CF','FC Internazionale Milano']
equipos_visitante = ['SSC Napoli', 'Tottenham Hotspur', 'FC Porto', 'SL Benfica', 'Real Madrid CF', 'Chelsea FC', 'FC Bayern München', 'Manchester City FC', 'AC Milan','SL Benfica','Chelsea FC','Manchester City FC','FC Internazionale Milano','Manchester City FC','Manchester City FC']
resultados = [2,1,1,2,2,2,2,2,2,1,1,2,2,2,2]

df = crear_datos(equipos_local, equipos_visitante, resultados)

#21-22
equipos_local = ['SL Benfica', 'FC Internazionale Milano', 'Villarreal CF', 'FC Salzburg', 'Manchester City FC', 'Club Atlético de Madrid', 'Chelsea FC', 'Paris Saint-Germain','SL Benfica','Villarreal CF','Manchester City FC','Chelsea FC','Liverpool FC', 'Manchester City FC','Liverpool FC']
equipos_visitante = ['Rabat Ajax FC', 'Liverpool FC', 'Juventus', 'FC Bayern München','Sporting Clube de Portugal', 'Manchester United', 'LOSC Lille','Real Madrid CF','Liverpool FC','FC Bayern München','Club Atlético de Madrid','Real Madrid CF','Villarreal CF','Real Madrid CF','Real Madrid CF']
resultados = [1,2,1,2,1,1,1,2,2,1,1,2,1,2,2]

df = agregar_datos(df, equipos_local, equipos_visitante, resultados)

#20-21
equipos_local = ['SS Lazio','Paris Saint-Germain','VfL Borussia Mönchengladbach', 'Borussia Dortmund', 'Real Madrid CF', 'Liverpool FC', 'FC Porto','Chelsea FC','FC Bayern München','Manchester City FC','Real Madrid CF','FC Porto','Paris Saint-Germain','Real Madrid CF','Manchester City FC']
equipos_visitante = ['FC Bayern München', 'FC Barcelona', 'Manchester City FC','Sevilla FC','Atalanta BC','RB Leipzig','Juventus','Club Atlético de Madrid', 'Paris Saint-Germain','Borussia Dortmund','Liverpool FC','Chelsea FC','Manchester City FC','Chelsea FC','Chelsea FC']
resultados = [2,1,2,1,1,1,1,1,2,1,1,2,2,2,2]

df = agregar_datos(df, equipos_local, equipos_visitante, resultados)

#19-20
equipos_local = ['Tottenham Hotspur', 'Club Atlético de Madrid', 'Atalanta BC', 'Borussia Dortmund', 'Real Madrid CF', 'Olympique Lyonnais', 'SSC Napoli', 'Chelsea FC', 'RB Leipzig', 'Atalanta BC', 'Manchester City FC', 'FC Barcelona', 'RB Leipzig', 'Olympique Lyonnais', 'Paris Saint-Germain']
equipos_visitante = ['RB Leipzig', 'Liverpool FC', 'Valencia CF', 'Paris Saint-Germain', 'Manchester City FC', 'Juventus', 'FC Barcelona', 'FC Bayern München', 'Club Atlético de Madrid', 'Paris Saint-Germain', 'Olympique Lyonnais', 'FC Bayern München', 'Paris Saint-Germain', 'FC Bayern München', 'FC Bayern München']
resultados = [2,1,1,2,2,2,1,2,2,1,2,2,2,2,2,2]

df = agregar_datos(df, equipos_local, equipos_visitante, resultados)

#18-19
equipos_local = ['Tottenham Hotspur', 'FC Schalke 04', 'Rabat Ajax FC', 'Club Atlético de Madrid', 'Manchester United', 'Olympique Lyonnais', 'Liverpool FC', 'AS Roma', 'Tottenham Hotspur', 'Rabat Ajax FC', 'Manchester United', 'Liverpool FC', 'Tottenham Hotspur', 'FC Barcelona', 'Liverpool FC']
equipos_visitante = ['Borussia Dortmund', 'Manchester City FC', 'Real Madrid CF', 'Juventus', 'Paris Saint-Germain', 'FC Barcelona', 'FC Bayern München', 'FC Porto', 'Manchester City FC', 'Juventus', 'FC Barcelona', 'FC Porto', 'Rabat Ajax FC', 'Liverpool FC', 'Liverpool FC']
resultados = [1,2,1,2,1,2,1,2,1,1,2,1,1,2,2]

df = agregar_datos(df, equipos_local, equipos_visitante, resultados)

#17-18
equipos_local = ['Sevilla FC', 'FC Bayern München', 'Juventus', 'Real Madrid CF', 'FC Porto', 'FC Basel 1893', 'Chelsea FC', 'FC Shakhtar Donetsk', 'Sevilla FC', 'Juventus', 'Liverpool FC', 'FC Barcelona', 'FC Bayern München', 'Liverpool FC','Real Madrid CF']
equipos_visitante = ['Manchester United', 'Beşiktaş JK', 'Tottenham Hotspur', 'Paris Saint-Germain', 'Liverpool FC', 'Manchester City FC', 'FC Barcelona', 'AS Roma', 'FC Bayern München', 'Real Madrid CF', 'Manchester City FC', 'AS Roma', 'Real Madrid CF', 'AS Roma', 'Liverpool FC']
resultados = [1,1,1,1,2,2,2,2,2,2,1,2,2,1,1]

df = agregar_datos(df, equipos_local, equipos_visitante, resultados)

#16-17
equipos_local = ['SL Benfica', 'Manchester City FC', 'FC Porto', 'Paris Saint-Germain', 'FC Bayern München', 'Real Madrid CF','Bayer 04 Leverkusen', 'Sevilla FC', 'Borussia Dortmund', 'Juventus', 'FC Bayern München', 'Club Atlético de Madrid', 'AS Monaco FC', 'Club Atlético de Madrid', 'Juventus']
equipos_visitante = ['Borussia Dortmund', 'AS Monaco FC', 'Juventus', 'FC Barcelona','Arsenal FC', 'SSC Napoli', 'Club Atlético de Madrid', 'Leicester City FC', 'AS Monaco FC', 'FC Barcelona', 'Real Madrid CF', 'Leicester City FC', 'Juventus', 'Club Atlético de Madrid', 'Real Madrid CF']
resultados = [2,2,2,2,1,1,1,2,2,2,1,2,1,2,1,2]

df = agregar_datos(df, equipos_local, equipos_visitante, resultados)

#15-16
equipos_local = ['Paris Saint-Germain', 'FC Dynamo Kyiv', 'KAA Gent', 'AS Roma', 'Arsenal FC', 'PSV Eindhoven', 'Juventus', 'SL Benfica','Paris Saint-Germain','VfL Wolfsburg', 'FC Barcelona', 'FC Bayern München', 'Manchester City FC', 'Club Atlético de Madrid', 'Real Madrid CF']
equipos_visitante = ['Chelsea FC', 'Manchester City FC', 'VfL Wolfsburg', 'Real Madrid CF', 'FC Barcelona', 'Club Atlético de Madrid', 'FC Bayern München', 'FC Zenit', 'Manchester City FC', 'Real Madrid CF', 'Club Atlético de Madrid', 'SL Benfica', 'Real Madrid CF', 'FC Bayern München', 'Club Atlético de Madrid']
resultados = [1,2,2,2,2,2,2,1,2,2,2,1,2,1,1]

df = agregar_datos(df, equipos_local, equipos_visitante, resultados)

#14-15
equipos_local = []
equipos_visitante = []
resultados = []

#13-14
equipos_local = []
equipos_visitante = []
resultados = []

#12-13
equipos_local = []
equipos_visitante = []
resultados = []

#11-12
equipos_local = []
equipos_visitante = []
resultados = []

#10-11
equipos_local = []
equipos_visitante = []
resultados = []

#09-10
equipos_local = []
equipos_visitante = []
resultados = []

#08-09
equipos_local = []
equipos_visitante = []
resultados = []




#lo pasamos a csv
df.to_csv('csvs/partidos_fut_dnn.csv', sep=';', index=False, encoding='utf-8')