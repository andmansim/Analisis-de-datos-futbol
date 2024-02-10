'''
Ejercicio de verdad
1. Encontrar las estadísticas de uefa de los años anteriores, para poder procesar los datos.
   Estas deben de tener PP, PE, PG, PJ, Puntos, Equipo. 
2. Necesito la lista de los equpos que van a jugar en este año
3. Puede que la lista de los partidos de los años anterirores
4. Ns que más. 
5. Hacer un master en fútbol. 
6. hacer gráficos de todos los años con las estadísticas de los últimos 10 años. 
7. usar la sopa para extraer datos de webs (en cripto 9022024)
'''

'''
¿Cómo funciona el fútbol?
-Grupos: Los clubs de fútbol se reparten en 8 grupos de 4 equipos cada uno. Cada equipo juega 6 partidos, 
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

import requests
from bs4 import BeautifulSoup

# URL del endpoint de la API de SofaScore que deseas acceder
url = 'https://api.sofascore.com/api/v1/your/endpoint'

# Tu clave de API de SofaScore
api_key = 'Bearer'

# Encabezado de autorización con la clave de API
headers = {
    'Authorization': f'Bearer {api_key}'
}

# Realiza una solicitud HTTP GET a la API de SofaScore
response = requests.get(url, headers=headers)

# Verifica si la solicitud fue exitosa (código de estado 200)
if response.status_code == 200:
    # Imprime la respuesta JSON
    data = response.json()
    print(data)
else:
    print("Error al obtener los datos:", response.status_code)
