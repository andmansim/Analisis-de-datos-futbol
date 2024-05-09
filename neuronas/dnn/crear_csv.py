import pandas as pd

#Vamos a crear un csv con distintos partidos y sus resultados
#para poder entrenar la red neuronal

#columnas
columnas = ['local', 'visitante', 'porganarpartido_local','porganarpartido_visitante','porperderpartido_local', 'porperderpartido_visitante', 'porcapacidad_ofensiva_local','porcapacidad_ofensiva_visitante', 'porcapacidad_defensiva_local','porcapacidad_defensiva_visitante', 'resultado']
#cogemos los datos de datos_fut.csv
df_equipos = pd.read_csv('csvs/datos_fut.csv', encoding='utf-8', sep=';')

#creamos un dataframe con las columnas que hemos creado
df = pd.DataFrame(columns = columnas)
#función que se encarga de recoger los datos de datos_fut.csv y añadirlo a la columna que corresponda
def recoger_datos(df, equipos_local, equipos_visitante, resultados):
    #asiganmos los valores a las columnas
    for i in range(len(equipos_local)):
        df['local'] = equipos_local[i]
        df['visitante'] = equipos_visitante[i]
        df['porganarpartido_local'] = df_equipos[df_equipos['club'] == equipos_local[i]]['porganarpartido'].values[0]
        df['porganarpartido_visitante'] = df_equipos[df_equipos['club'] == equipos_visitante[i]]['porganarpartido'].values[0]
        df['porperderpartido_local'] = df_equipos[df_equipos['club'] == equipos_local[i]]['porperderpartido'].values[0]
        df['porperderpartido_visitante'] = df_equipos[df_equipos['club'] == equipos_visitante[i]]['porperderpartido'].values[0]
        df['porcapacidad_ofensiva_local'] = df_equipos[df_equipos['club'] == equipos_local[i]]['porcapacidad_ofensiva'].values[0]
        df['porcapacidad_ofensiva_visitante'] = df_equipos[df_equipos['club'] == equipos_visitante[i]]['porcapacidad_ofensiva'].values[0]
        df['porcapacidad_defensiva_local'] = df_equipos[df_equipos['club'] == equipos_local[i]]['porcapacidad_defensiva'].values[0]
        df['porcapacidad_defensiva_visitante'] = df_equipos[df_equipos['club'] == equipos_visitante[i]]['porcapacidad_defensiva'].values[0]
        df['resultado'] = resultados[i]
        #añadimos la fila al dataframe
        df = df.append(df, ignore_index=True)
    #lo pasamos a csv
    df.to_csv('csvs/partidos_fut.csv', sep=';', index=False, encoding='utf-8')
    

#dividimos los equpos en dos listas, local y visitante 
#22-23
equipos_local = ['Eintracht Frankfurt', 'AC Milan','FC Internazionale Milano','Club Brugge','Liverpool FC', 'Borussia Dortmund', 'Paris Saint-Germain', 'RB Leipzig', 'SS Napoli', 'FC Internazionale Milano', 'Real Madrid CF','FC Bayern MÃ¼nchen','AC Milan','Real Madrid CF','FC Internazionale Milano']
equipos_visitante = ['SSC Napoli', 'Tottenham Hotspur', 'FC Porto', 'SL Benfica', 'Real Madrid CF', 'Chelsea FC', 'FC Bayern MÃ¼nchen', 'Manchester City FC', 'AC Milan','SL Benfica','Chelsea FC','Manchester City FC','FC Internazionale Milano','Manchester City FC','Manchester City FC']
resultados = [2,1,1,2,2,2,2,2,2,1,1,2,2,2,2]
recoger_datos(df_equipos, equipos_local, equipos_visitante, resultados)

#21-22
equipos_local = ['SL Benfica', 'FC Internazionale Milano', 'Villarreal CF', 'FC Salzburg', 'Manchester City FC', 'Club AtlÃ©tico de Madrid', 'Chelsea FC', 'Paris Saint-Germain','SL Benfica','Villarreal CF','Manchester City FC','Chelsea FC','Liverpool FC', 'Manchester City FC','Liverpool FC']
equipos_visitante = ['Rabat Ajax FC', 'Liverpool FC', 'Juventus', 'FC Bayern MÃ¼nchen','Sporting Clube de Portugal', 'Manchester United', 'LOSC Lille','Real Madrid CF','Liverpool FC','FC Bayern MÃ¼nchen','Club AtlÃ©tico de Madrid','Real Madrid CF','Villarreal CF','Real Madrid CF','Real Madrid CF']
resultados = [1,2,1,2,1,1,1,2,2,1,1,2,1,2,2]

#20-21
equipos_local = ['SS Lazio','Paris Saint-Germain','VfL Borussia MÃ¶nchengladbach', 'Borussia Dortmund', 'Real Madrid CF', 'Liverpool FC', 'FC Porto','Chelsea FC','FC Bayern MÃ¼nchen','Manchester City FC','Real Madrid CF','FC Porto','Paris Saint-Germain','Real Madrid CF','Manchester City FC']
equipos_visitante = ['FC Bayern MÃ¼nchen', 'FC Barcelona', 'Manchester City FC','Sevilla FC','Atalanta BC','RB Leipzig','Juventus','Club AtlÃ©tico de Madrid', 'Paris Saint-Germain','Borussia Dortmund','Liverpool FC','Chelsea FC','Manchester City FC','Chelsea FC','Chelsea FC']
resultados = [2,1,2,1,1,1,1,1,2,1,1,2,2,2,2]

#19-20
equipos_local = ['Tottenham Hotspur', 'Club AtlÃ©tico de Madrid', 'Atalanta BC', 'Borussia Dortmund', 'Real Madrid CF', 'Olympique Lyonnais', 'SS Napoli', 'Chelsea FC', 'RB Leipzig', 'Atalanta BC', 'Manchester City FC', 'FC Barcelona', 'RB Leipzig', 'Olympique Lyonnais', 'Paris Saint-Germain']
equipos_visitante = ['RB Leipzig', 'Liverpool FC', 'Valencia CF', 'Paris Saint-Germain', 'Manchester City FC', 'Juventus', 'FC Barcelona', 'FC Bayern MÃ¼nchen', 'Club AtlÃ©tico de Madrid', 'Paris Saint-Germain', 'Olympique Lyonnais', 'FC Bayern MÃ¼nchen', 'Paris Saint-Germain', 'FC Bayern MÃ¼nchen', 'FC Bayern MÃ¼nchen']
resultados = [2,1,1,2,2,2,1,2,2,1,2,2,2,2,2,2]

#18-19
equipos_local = ['Totenham Hotspur', 'FC Schalke 04', 'Rabat Ajax FC', 'Club AtlÃ©tico de Madrid', 'Manchester United', 'Olimpique Lyonnais', 'Liverpool FC', 'AS Roma', 'Tottenham Hotspur', 'Rabat Ajax FC', 'Macnhester United', 'Liverpool FC', 'Tottenham Hotspur', 'FC Barcelona', 'Liverpool FC']
equipos_visitante = ['Borussia Dortmund', 'Manchester City FC', 'Real Madrid CF', 'Juventus', 'Paris Saint-Germain', 'FC Barcelona', 'FC Bayern MÃ¼nchen', 'FC Porto', 'Manchester City FC', 'Juventus', 'FC Barcelona', 'FC Porto', 'Rabat Ajax FC', 'Liverpool FC', 'Liverpool FC']
resultados = [1,2,1,2,1,2,1,2,1,1,2,1,1,2,2]

#17-18
equipos_local = ['Sevilla FC', 'FC Bayern MÃ¼nchen', 'Juventus', 'Real Madrid CF', 'FC Porto', 'FC Basel 1893', 'Chelsea FC', 'FC Shakhtar Donetsk', 'Sevilla FC', 'Juventus', 'Liverpool FC', 'FC Barcelona', 'FC Bayern MÃ¼nchen', 'Liverpool FC','Real Madrid CF']
equipos_visitante = ['Manchester United', 'Besiktas JK', 'Tottenham Hotspur', 'Paris Saint-Germain', 'Liverpool FC', 'Manchester City FC', 'FC Barcelona', 'AS Roma', 'FC Bayern MÃ¼nchen', 'Real Madrid CF', 'Manchester City FC', 'AS Roma', 'Real Madrid CF', 'AS Roma', 'Liverpool FC']
resultados = [1,1,1,1,2,2,2,2,2,2,1,2,2,1,1]

#16-17
equipos_local = ['SL Benfica', 'Manchester City FC', 'FC Porto', 'Paris Saint-Germain', 'FC Bayern MÃ¼nchen', 'Real Madrid CF','Bayer 04 Leverkusen', 'Sevilla FC', 'Borussia Dortmund', 'Juventus', 'FC Bayern MÃ¼nchen', 'Club AtlÃ©tico de Madrid', 'AS Monaco FC', 'Club AtlÃ©tico de Madrid', 'Juventus']
equipos_visitante = ['Borussia Dortmund', 'AS Monaco FC', 'Juventus', 'FC Barcelona','Arsenal FC', 'SSC Napoli', 'Club AtlÃ©tico de Madrid', 'Leicester City FC', 'AS Monaco FC', 'FC Barcelona', 'Real Madrid CF', 'Leicester City FC', 'Juventus', 'Club AtlÃ©tico de Madrid', 'Real Madrid CF']
resultados = [2,2,2,2,1,1,1,2,2,2,1,2,1,2,1,2]

#15-16
equipos_local = ['Paris Saint-Germain', 'FC Dynamo Kyiv', 'KAA Gent', 'AS Roma', 'Arsenal FC', 'PSV Eindhoven', 'Juventus', 'SL Benfica','Paris Saint-Germain','VfL Wolfsburg', 'FC Barcelona', 'FC Bayern MÃ¼nchen', 'Manchester City FC', 'Club AtlÃ©tico de Madrid', 'Real Madrid CF']
equipos_visitante = ['Chelsea FC', 'Manchester City FC', 'VfL Wolfsburg', 'Real Madrid CF', 'FC Barcelona', 'Club AtlÃ©tico de Madrid', 'FC Bayern MÃ¼nchen', 'FC Zenit', 'Manchester City FC', 'Real Madrid CF', 'Club AtlÃ©tico de Madrid', 'SL Benfica', 'Real Madrid CF', 'FC Bayern MÃ¼nchen', 'Club AtlÃ©tico de Madrid']
resultados = [1,2,2,2,2,2,2,1,2,2,2,1,2,1,1]

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




