# Metodo Grafico
# Algoritmo de Optimizacion
# by: Alex, Matias
# UCT

EPS_DET    = 1e-10   
EPS_COORD  = 1e-5    
MARGEN     = 1e-4    
ROUND_DIG  = 4       
MIN_RANGO  = 10      # Valor mínimo de referencia cuando no hay intersecciones grandes
ESCALA_EJE = 1.1     # Factor de expansión de los límites de los ejes
MARGEN_NEG = 0.05    # Fracción de margen negativo en los ejes (origen visual)
N_PUNTOS_X = 400     # Resolución del array x para graficar rectas

# Gráfico 
FIG_W      = 8       
FIG_H      = 6       
MS_OPTIMO  = 10      # Tamaño del marcador del punto óptimo
MS_VERTICE = 4       # Tamaño de los marcadores de vértices
MS_REGION  = 8       # Tamaño del marcador cuando la región es un solo punto
LW_REGION  = 1       # Grosor del contorno del hull convexo
LW_LINEA   = 3       # Grosor de la región factible cuando es un segmento
LW_FLECHA  = 2       # Grosor de las flechas de región no acotada
PAUSA_SEG  = 0.8     # Pausa entre restricciones al graficar (segundos)

ESCALA_HULL   = 0.95  # Fracción del límite del eje usada para puntos extra del hull
FLECHA_INICIO = 0.72  # Fracción del eje donde comienza la flecha
FLECHA_FIN    = 0.98  # Fracción del eje donde termina la flecha
FLECHA_TRANS  = 0.25  # Fracción transversal fija de la flecha
EXTRA_BORDE1  = 0.1   # Primera fracción interior para puntos extra del hull
EXTRA_BORDE2  = 0.5   # Segunda fracción interior para puntos extra del hull

FONTSIZE       = 8
ALPHA_REGION   = 0.3
ALPHA_GRID     = 0.4
ALPHA_ANOTACION = 0.5
COLOR_REGION   = 'skyblue'
COLOR_HULL     = 'blue'
COLOR_GRID_H   = 'black'
COLOR_OPTIMO   = 'ro'          
COLOR_VERTICE  = 'ko'
COLOR_ANOTACION = 'yellow'
COLOR_FLECHA   = 'blue'


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from itertools import combinations

# Formato de los numeros: Muestra enteros sin decimales y flotantes con dos decimales
def formato_numero(num):
    return f"{int(num)}" if num == int(num) else f"{num:.2f}"

#funcion que solicita al usuario ingresar los datos minimos necesarios para graficar las rectas.
def ingresar_datos():
    n = int(input("Numero de restricciones: "))
    Matriz, Restriccion = [],[]
    for i in range(n):
        print(f"\nIngrese los datos para las restricciones {i+1}: ")
        a = float(input("Coeficiente de x(a): "))
        b = float(input("Coeficiente de y(b): "))
        
        Tipo_Restriccion = input("Tipo de restricción (<=, >=, =): ").strip()
        while Tipo_Restriccion not in ['<=', '>=', '=']: 
            Tipo_Restriccion = input("Tipo invalido. Ingrese <=, >= o =: ").strip()
        
        c = float(input("Termino Constante (c): "))
        Matriz.append([a,b,c])
        Restriccion.append(Tipo_Restriccion)
        
    print("\nIngrese los coeficientes de la funcion Objetivo Z:")
    Coeficiente_x = float(input("Coeficiente de x: "))
    Coeficiente_y = float(input("Coeficiente de y: "))
    
    print("\n Que desea hacer con la funcion objetivo?\n1. Maximizar\n2. Minimizar")
    tipo_funcion = 'max' if input("Seleccione una opcion 1 o 2: ").strip() == '1' else 'min'
    
    return np.array(Matriz), Restriccion, (Coeficiente_x,Coeficiente_y), tipo_funcion
    # el return devuelve una tupla con esos 4 elementos

# Muestra el sistema de restricciones ingresado por el usuario
def mostrar_ecuaciones(Matriz, Restriccion):
    print("\n--- Restricciones ---")
    for i, (a, b, c) in enumerate(Matriz):
        signo = "+" if b >= 0 else "-"
        print(f"Restriccion {i+1}: {formato_numero(a)}x {signo} {formato_numero(abs(b))}y {Restriccion[i]} {formato_numero(c)}")
    print("Condición de no negatividad: x >= 0, y >= 0")

# Calcular las intersecciones entre todas las rectas y los ejes usando la Regla de Cramer, combina todos los pares posibles de ecuaciones para encontrar sus puntos de cruce.
def calcular_intersecciones(Matriz):
    print("\n --- Puntos de Interseccion ---")
    intersecciones = []
    ejes = np.array([
    [1, 0, 0],  # x = 0  (eje Y)
    [0, 1, 0]   # y = 0  (eje X)
])
    todas_rectas = np.vstack((Matriz,ejes))
    
    for r1,r2 in combinations(todas_rectas,2):
        determinante = r1[0] * r2[1] - r2[0] * r1[1]
        if abs(determinante) > EPS_DET:
            x = (r1[2] * r2[1] - r2[2] * r1[1]) / determinante # calculo de coordenada x
            y = (r1[0] * r2[2] - r2[0] * r1[2]) / determinante # calculo de coordenada y
            
            if x >= -EPS_COORD and y >= -EPS_COORD:
                intersecciones.append((round(max(x, 0), 10), round(max(y, 0), 10))) # guardar el punto 
                
    intersecciones.append((0.0, 0.0)) # origen
    inter_unicas = list(set((round(x, 4), round(y, 4)) for x, y in intersecciones))
    for x, y in inter_unicas:
        print(f"Intersección: ({formato_numero(x)}, {formato_numero(y)})")                       
    return inter_unicas

def evaluar_optimo(optimos, objetivo_funcion, tipo_funcion, ax):
    c_x, c_y = objetivo_funcion
    mejor_z, mejor_pt = None,None # no encontrados inicialmente
    for x, y in optimos: # el optimo es un problema lineal con region factible acotada, está en uno de los vertices. Evaluamos todos
        z = c_x * x + c_y * y
        print(f"Z({formato_numero(x)}, {formato_numero(y)}) = {formato_numero(z)}") # ver el valor de z en cada vertice 
        
        if mejor_z is None or (tipo_funcion == 'max' and z > mejor_z) or (tipo_funcion == 'min' and z < mejor_z): # decide si el punto actual es mejor que el mejor encontrado hasta ahora.
            mejor_z = z
            mejor_pt = (x, y)
    
    if mejor_pt: # comprueba que se encontró al menos un punto optimo 
        x_o, y_o = mejor_pt
        tipo_texto = "Máximo" if tipo_funcion == 'max' else "Mínimo"
        print(f"\nPunto óptimo ({tipo_texto}): ({formato_numero(x_o)}, {formato_numero(y_o)}) con Z = {formato_numero(mejor_z)}")
        
        ax.plot(
            x_o, y_o, COLOR_OPTIMO,
            markersize=MS_OPTIMO,
            label=f'Solución Óptima ({tipo_texto})',
            zorder=10
        )
        ax.annotate(
            f"SOLUCIÓN\n({formato_numero(x_o)}, {formato_numero(y_o)})\nZ={formato_numero(mejor_z)}",
            (x_o, y_o),
            xytext=(15, 15),
            textcoords='offset points',
            bbox=dict(boxstyle="round,pad=0.3", fc=COLOR_ANOTACION, alpha=ALPHA_ANOTACION),
            fontweight='bold'
        )




# Grafica el sistema de inecuaciones, pinta el area factible y llama a la funcion de evaluacion     
def graficar_y_resolver(Matriz, Restriccion, intersecciones, obj, tipo):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_title("Método Gráfico (Restricciones)")
    ax.grid(True, linestyle='--', alpha=ALPHA_GRID)
    ax.axhline(0, color=COLOR_GRID_H, linewidth=0.8)
    ax.axvline(0, color=COLOR_GRID_H, linewidth=0.8)
    
    # Límites
    tiene_mayor_igual = any(t == '>=' for t in Restriccion)
    max_x = max([x for x, y in intersecciones] + [MIN_RANGO]) * ESCALA_EJE
    max_y = max([y for x, y in intersecciones] + [MIN_RANGO]) * ESCALA_EJE
    ax.set_xlim(-max_x * MARGEN_NEG, max_x)
    ax.set_ylim(-max_y * MARGEN_NEG, max_y)
    x_vals = np.linspace(-max_x * MARGEN_NEG, max_x, N_PUNTOS_X)
    
    # Graficar rectas
    plt.ion()
    colores = plt.colormaps['tab10'].colors
    for i, (a, b, c) in enumerate(Matriz):
        color = colores[i % len(colores)]
        if b != 0:
            ax.plot(x_vals, (c - a * x_vals) / b, color=color,
                    label=f"R{i+1}: {formato_numero(a)}x + {formato_numero(b)}y = {formato_numero(c)}")
        elif a != 0:
            ax.axvline(x=c/a, color=color, label=f"R{i+1}: x = {formato_numero(c/a)}")
        ax.legend(fontsize=FONTSIZE, loc='upper right')
        plt.pause(PAUSA_SEG)
    plt.ioff()
    
    # Filtrar puntos factibles
    optimos = []
    for x, y in intersecciones:
        if all((t == '<=' and a*x + b*y <= c + MARGEN) or
               (t == '>=' and a*x + b*y >= c - MARGEN) or
               (t == '=' and abs(a*x + b*y - c) <= MARGEN)
               for (a, b, c), t in zip(Matriz, Restriccion)):
            optimos.append((round(x, ROUND_DIG), round(y, ROUND_DIG)))
    
    optimos = list(set(optimos))
    if not optimos:
        print("\nNo hay región factible.")
        plt.show()
        return
    
    # Dibujar región factible
    region_no_acotada = False
    if len(optimos) > 2:
        try:
            puntos = np.array(optimos)
            if tiene_mayor_igual:
                lim_x, lim_y = max_x * ESCALA_HULL, max_y * ESCALA_HULL
                extras = [[lim_x, 0], [0, lim_y], [lim_x, lim_y],
                          [lim_x, lim_y * EXTRA_BORDE2], [lim_x * EXTRA_BORDE2, lim_y],
                          [lim_x * EXTRA_BORDE1, lim_y], [lim_x, lim_y * EXTRA_BORDE1]]
                for px, py in extras:
                    if all((t == '<=' and a*px + b*py <= c + MARGEN) or
                           (t == '>=' and a*px + b*py >= c - MARGEN) or
                           (t == '=' and abs(a*px + b*py - c) <= MARGEN)
                           for (a, b, c), t in zip(Matriz, Restriccion)):
                        puntos = np.vstack([puntos, [px, py]])
                        region_no_acotada = True
            hull = ConvexHull(puntos)
            verts = puntos[hull.vertices]
            ax.fill(verts[:, 0], verts[:, 1], alpha=ALPHA_REGION, color=COLOR_REGION)
            ax.plot(np.append(verts[:, 0], verts[0, 0]),
                    np.append(verts[:, 1], verts[0, 1]), COLOR_HULL, lw=LW_REGION)
        except:
            ax.plot([p[0] for p in optimos], [p[1] for p in optimos], 'b-', lw=LW_LINEA)
    elif len(optimos) == 2:
        ax.plot([p[0] for p in optimos], [p[1] for p in optimos], 'b-', lw=LW_LINEA)
    else:
        ax.plot(optimos[0][0], optimos[0][1], 'bo', markersize=MS_REGION)
    
    # Flechas región no acotada
    if region_no_acotada:
        ax.annotate('', xy=(max_x * FLECHA_FIN, max_y * FLECHA_TRANS),
                    xytext=(max_x * FLECHA_INICIO, max_y * FLECHA_TRANS),
                    arrowprops=dict(arrowstyle='->', color=COLOR_FLECHA, lw=LW_FLECHA))
        ax.annotate('', xy=(max_x * FLECHA_TRANS, max_y * FLECHA_FIN),
                    xytext=(max_x * FLECHA_TRANS, max_y * FLECHA_INICIO),
                    arrowprops=dict(arrowstyle='->', color=COLOR_FLECHA, lw=LW_FLECHA))
    
    # Vértices y óptimo
    for x, y in intersecciones:
        ax.plot(x, y, COLOR_VERTICE, markersize=MS_VERTICE, zorder=5)
    
    evaluar_optimo(optimos, obj, tipo, ax)
    ax.legend(fontsize=FONTSIZE, loc='upper right')
    plt.show()

def iniciar_app():
    """Función principal que coordina el flujo completo del algoritmo."""
    Matriz, Restriccion, obj, tipo_funcion = ingresar_datos()
    mostrar_ecuaciones(Matriz, Restriccion)
    intersecciones = calcular_intersecciones(Matriz)
    graficar_y_resolver(Matriz, Restriccion, intersecciones, obj, tipo_funcion)

if __name__ == "__main__":
    iniciar_app()
