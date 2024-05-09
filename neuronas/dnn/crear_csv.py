import pandas as pd

#Vamos a crear un csv con distintos partidos y sus resultados
#para poder entrenar la red neuronal

#columnas
columnas = ['local', 'visitante', 'porganarpartido_local','porganarpartido_visitante','porperderpartido_local', 'porperderpartido_visitante', 'porcapacidad_ofensiva_local','porcapacidad_ofensiva_visitante', 'porcapacidad_defensiva_local','porcapacidad_defensiva_visitante', 'resultado']

#función que se encarga de recoger los datos de datos_fut.csv y añadirlo a la columna que corresponda
def recoger_datos(data, equipos, ronda):
    pass

#dividimos los equpos en dos listas, local y visitante 
#22-23
equipos_local = ['Eintracht Frankfurt', 'AC Milan','FC Internazionale Milano','Club Brugge','Liverpool FC', 'Borussia Dortmund', 'Paris Saint-Germain', 'RB Leipzig', 'SS Napoli', 'FC Internazionale Milano', 'Real Madrid CF','FC Bayern MÃ¼nchen','AC Milan','Real Madrid CF','FC Internazionale Milano']
equipos_visitante = ['SSC Napoli', 'Tottenham Hotspur', 'FC Porto', 'SL Benfica', 'Real Madrid CF', 'Chelsea FC', 'FC Bayern MÃ¼nchen', 'Manchester City FC', 'AC Milan','SL Benfica','Chelsea FC','Manchester City FC','FC Internazionale Milano','Manchester City FC','Manchester City FC']
resultados = [2,1,1,2,2,2,2,2,2,1,1,2,2,2,2]
#21-22
equipos_local = ['SL Benfica', 'FC Internazionale Milano', 'Villarreal CF', 'FC Salzburg', 'Manchester City FC', 'Club AtlÃ©tico de Madrid', 'Chelsea FC', 'Paris Saint-Germain','SL Benfica','Villarreal CF','Manchester City FC','Chelsea FC','Liverpool FC', 'Manchester City FC','Liverpool FC']
equipos_visitante = ['Rabat Ajax FC', 'Liverpool FC', 'Juventus', 'FC Bayern MÃ¼nchen','Sporting Clube de Portugal', 'Manchester United', 'LOSC Lille','Real Madrid CF','Liverpool FC','FC Bayern MÃ¼nchen','Club AtlÃ©tico de Madrid','Real Madrid CF','Villarreal CF','Real Madrid CF','Real Madrid CF']
resultados = [1,2,1,2,1,1,1,2,2,1,1,2,1,2,2]

#20-21
equipos_local = ['SS Lazio','Paris Saint-Germain','VfL Borussia MÃ¶nchengladbach', 'Borussia Dortmund', 'Real Madrid CF', 'Liverpool FC', 'FC Porto','Chelsea FC','FC Bayern MÃ¼nchen','Manchester City FC','Real Madrid CF','FC Porto','Paris Saint-Germain','Real Madrid CF','Manchester City FC']
equipos_visitante = ['FC Bayern MÃ¼nchen', 'FC Barcelona', 'Manchester City FC','Sevilla FC','Atalanta BC','RB Leipzig','Juventus','Club AtlÃ©tico de Madrid', 'Paris Saint-Germain','Borussia Dortmund','Liverpool FC','Chelsea FC','Manchester City FC','Chelsea FC','Chelsea FC']
resultados = [2,1,2,1,1,1,1,1,2,1,1,2,2,2,2]

#



