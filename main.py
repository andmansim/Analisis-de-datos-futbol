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

import requests
from bs4 import BeautifulSoup

# URL de la página web que quieres hacer scraping
url = 'https://www.sofascore.com/tournament/football/europe/uefa-champions-league/7#23766'

# Configurar el encabezado "User-Agent" para simular un navegador web real
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# Hacer una solicitud HTTP a la página web con el encabezado definido
response = requests.get(url, headers=headers)

# Crear un objeto BeautifulSoup con el contenido de la página web
soup = BeautifulSoup(response.text, 'lxml')

# Obtener el título de la página web
titulo = soup.title.text.strip()
print("Título de la página:", titulo)

# Buscar un elemento específico en el HTML (si es necesario)
elementos = soup.find_all('div', {'class': 'sc-fqkvVR LZMLY'})
    
    
# Encontrar el contenedor principal
contenedor = soup.find('div', class_='sc-fqkvVR eeeBnr sc-d8bc48b6-2 eYMsrH')

# Iterar sobre cada subdivisión dentro del contenedor
for subdivision in contenedor.find_all('div', class_='sc-fqkvVR LZMLY'):
    # Encontrar cada dato dentro de la subdivisión y extraer su texto
    datos = [dato.get_text() for dato in subdivision.find_all('p')]
    
    # Imprimir los datos
    print(datos)
    
    
