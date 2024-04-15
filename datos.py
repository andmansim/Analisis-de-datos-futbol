from bs4 import BeautifulSoup
import requests

url = 'https://www.uefa.com/uefachampionsleague/statistics/players/?sortBy=minutes_played_official&order=aschttps://www.uefa.com/uefachampionsleague/statistics/players/?sortBy=minutes_played_official&order=asc'
response = requests.get(url)
html = response.text
soup = BeautifulSoup(html, 'lxml')

# Utiliza el selector CSS para encontrar la tabla
#tabla_jugadores = soup.select('body > div.main-wrap > div > div > div.content > div.d3-react.statistics-detail.pk-theme--light > div > div > div:nth-child(4) > div > div.HuCvdCPuILEJ2xfnIMRi > div > div.ag-root-wrapper.ag-ltr.ag-layout-auto-height > div.ag-root-wrapper-body.ag-focus-managed.ag-layout-auto-height > div.ag-root.ag-unselectable.ag-layout-auto-height > div.ag-body.ag-layout-auto-height > div.ag-body-viewport.ag-row-animation.ag-layout-auto-height')
tabla_jugadores  = soup.find_all(attrs={'role':'presentation'})
datos = soup.find_all(attrs={'role':'gridcell'})
for fila in tabla_jugadores:
    print(fila)
    nombre = fila.find_all('span', slot='primary')
    equipo = fila.find_all('span', slot='secondary')
    print(nombre)
    print(equipo)
    minutos = fila.find_all('span', class_='ag-cell--sorted')
    partidos = fila.find_all('span', class_='ag-cell-value')
    
'''for celda in datos:
    print(celda)
    minutos = celda.find_all()    
for fila in tabla_jugadores:
    print(fila)
    datos_jugador ={}
    nombre= fila.find_all('span', slot= 'primary')
    equipo= fila.find_all('span', slot= 'secondary')
    for celda in fila.find_all('td'):
        if 'nombre' not in datos_jugador:
            datos_jugador['nombre'] = celda.text.strip()
        elif 'equipo' not in datos_jugador:
            datos_jugador['equipo'] = celda.text.strip()
        elif 'minutos' not in datos_jugador:
            datos_jugador['minutos'] = celda.text.strip()
        elif 'goles' not in datos_jugador:
            datos_jugador['goles'] = celda.text.strip()
        elif 'asistencias' not in datos_jugador:
            datos_jugador['asistencias'] = celda.text.strip()
    print(datos_jugador)
'''