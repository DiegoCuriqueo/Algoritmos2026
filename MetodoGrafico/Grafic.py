# Metodo Grafico
# Algoritmo de Optimizacion
# by: Alex, Matias
# UCT

# Variables Globales
EPS_DET    = 1e-10   # su funcion principal es evitar los errores matematicos
EPS_COORD  = 1e-5    # evita errores de calculo
MARGEN     = 1e-4    # Margen de error para comparaciones de punto flotante.
ROUND_DIG  = 4       # Redondea los números a 4 decimales
MIN_RANGO  = 10      # Valor mínimo de referencia cuando no hay intersecciones grandes
ESCALA_EJE = 1.1     # Factor de expansión de los límites de los ejes
MARGEN_NEG = 0.05    # Fracción de margen negativo en los ejes (origen visual)
N_PUNTOS_X = 400     # Resolución del array x para graficar rectas

# Gráfico 
PAUSA_SEG  = 0.8     # Pausa entre restricciones al graficar (segundos)
ESCALA_HULL   = 0.95  # Fracción del límite del eje usada para puntos extra del hull
COLOR_ANOTACION = 'yellow' 
COLOR_FLECHA   = 'blue' 

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from itertools import combinations

# Formato de los numeros: Muestra enteros sin decimales y flotantes con dos decimales
def formato_numero(num):
    return f"{int(num)}" if num == int(num) else f"{num:.2f}"

# Funciones de lectura de datos
def leer_entero(mensaje):
    while True:
        try:
            valor = int(input(mensaje))
            if valor <= 0:
                print(" Error: Ingrese un número entero positivo mayor a 0.")
                continue
            return valor
        except ValueError:
            print(" Error: Entrada inválida. Debe ingresar un número entero.")

def leer_flotante(mensaje):
    while True:
        try:
            return float(input(mensaje))
        except ValueError:
            print(" Error: Entrada inválida. Debe ingresar un número válido.")
            

#funcion que solicita al usuario ingresar los datos minimos necesarios para graficar las rectas.
def ingresar_datos():
    print("\nIngrese los coeficientes de la funcion Objetivo Z:")
    Coeficiente_x = leer_flotante("Coeficiente de x: ")
    Coeficiente_y = leer_flotante("Coeficiente de y: ")
    
    print("\n Que desea hacer con la funcion objetivo?\n1. Maximizar\n2. Minimizar")
    opcion = input("Seleccione una opcion 1 o 2: ").strip()
    while opcion not in ['1', '2']:
        opcion = input(" Error. Seleccione una opcion 1 (Maximizar) o 2 (Minimizar): ").strip()
        
    tipo_funcion = 'max' if opcion == '1' else 'min'

    print("\n--- Restricciones ---")
    n = leer_entero("Numero de restricciones: ")
    Matriz, Restriccion = [],[]
    for i in range(n):
        print(f"\nIngrese los datos para la restriccion {i+1}: ")
        a = leer_flotante("Coeficiente de x(a): ")
        b = leer_flotante("Coeficiente de y(b): ")
        
        Tipo_Restriccion = input("Tipo de restricción (<=, >=, =): ").strip()
        while Tipo_Restriccion not in ['<=', '>=', '=']: 
            Tipo_Restriccion = input(" Tipo invalido. Ingrese <=, >= o =: ").strip()
        
        c = leer_flotante("Termino Constante (c): ")
        Matriz.append([a,b,c])
        Restriccion.append(Tipo_Restriccion)
        
    return np.array(Matriz), Restriccion, (Coeficiente_x,Coeficiente_y), tipo_funcion


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
    inter_unicas = list(set((round(x, ROUND_DIG), round(y, ROUND_DIG)) for x, y in intersecciones))
    for x, y in inter_unicas:
        print(f"Intersección: ({formato_numero(x)}, {formato_numero(y)})")                       
    return inter_unicas


def evaluar_optimo(optimos, objetivo_funcion, tipo_funcion, ax):
    c_x, c_y = objetivo_funcion
    for x, y in optimos: # el optimo es un problema lineal con region factible acotada, está en uno de los vertices. Evaluamos todos
        print(f"Z({formato_numero(x)}, {formato_numero(y)}) = {formato_numero(c_x * x + c_y * y)}") # ver el valor de z en cada vertice 
        
    calc_z = lambda pt: c_x * pt[0] + c_y * pt[1]
    mejor_pt = max(optimos, key=calc_z) if tipo_funcion == 'max' else min(optimos, key=calc_z)
    mejor_z = calc_z(mejor_pt)
    
    x_o, y_o = mejor_pt
    tipo_texto = "Máximo" if tipo_funcion == 'max' else "Mínimo"
    print(f"\nPunto óptimo ({tipo_texto}): ({formato_numero(x_o)}, {formato_numero(y_o)}) con Z = {formato_numero(mejor_z)}")
    
    ax.plot(x_o, y_o, 'ro', markersize=10, label=f'Solución Óptima ({tipo_texto})', zorder=10)
    ax.annotate(
        f"SOLUCIÓN\n({formato_numero(x_o)}, {formato_numero(y_o)})\nZ={formato_numero(mejor_z)}",
        (x_o, y_o), xytext=(15, 15), textcoords='offset points',
        bbox=dict(boxstyle="round,pad=0.3", fc='yellow', alpha=0.5), fontweight='bold'
    )


# se encarga puramente de la logica matematica para filtrar y devolver soo los vertices que cumplen con todas las restricciones
def es_factible(x, y, Matriz, Restriccion):
    return all((t == '<=' and a*x + b*y <= c + MARGEN) or
               (t == '>=' and a*x + b*y >= c - MARGEN) or
               (t == '=' and abs(a*x + b*y - c) <= MARGEN)
               for (a, b, c), t in zip(Matriz, Restriccion))

# se encarga puramente de la logica matematica para filtrar y devolver soo los vertices que cumplen con todas las restricciones
def obtener_puntos_factibles(intersecciones, Matriz, Restriccion):
    return list(set((round(x, ROUND_DIG), round(y, ROUND_DIG)) for x, y in intersecciones if es_factible(x, y, Matriz, Restriccion)))


# Esta funcion se encarga de dibujar cada una de las rectas de la matriz, una por una de forma incremental.
def dibujar_lineas_restriccion(ax, Matriz, x_vals):
    plt.ion()
    colores = plt.colormaps['tab10'].colors
    for i, (a, b, c) in enumerate(Matriz):
        color = colores[i % len(colores)]
        if b != 0:
            ax.plot(x_vals, (c - a * x_vals) / b, color=color,
                    label=f"R{i+1}: {formato_numero(a)}x + {formato_numero(b)}y = {formato_numero(c)}")
        elif a != 0:
            ax.axvline(x=c/a, color=color, label=f"R{i+1}: x = {formato_numero(c/a)}")
        ax.legend(fontsize=8, loc='upper right')
        plt.pause(PAUSA_SEG)
    plt.ioff()


# Esta funcion se encarga de dibujar el poligono factible, considerando si la region es acotada o no.
def dibujar_poligono_factible(ax, optimos, Matriz, Restriccion, max_x, max_y):
    puntos = np.array(optimos) if len(optimos) > 0 else np.empty((0, 2))
    
    # Siempre verificamos puntos extra en los límites (para regiones no acotadas)
    lim_x, lim_y = max_x * ESCALA_HULL, max_y * ESCALA_HULL
    extras = [[lim_x, i] for i in np.linspace(0, lim_y, 100)] + \
             [[i, lim_y] for i in np.linspace(0, lim_x, 100)]
        
    puntos_top, puntos_right, valid_extras = [], [], []
    for px, py in extras:
        if es_factible(px, py, Matriz, Restriccion):
            valid_extras.append([px, py])
            if py == lim_y: puntos_top.append(px)
            if px == lim_x: puntos_right.append(py)
            
    if valid_extras:
        puntos = np.vstack([puntos, valid_extras]) if len(puntos) > 0 else np.array(valid_extras)
            
    if len(puntos) > 2:
        try:
            hull = ConvexHull(puntos)
            verts = puntos[hull.vertices]
            ax.fill(verts[:, 0], verts[:, 1], alpha=0.3, color='skyblue')
            ax.plot(np.append(verts[:, 0], verts[0, 0]),
                    np.append(verts[:, 1], verts[0, 1]), 'blue', lw=1)
        except:
            # Si ConvexHull falla (puntos colineales), es una región factible lineal (segmento o rayo)
            p_ordenados = puntos[np.lexsort((puntos[:,1], puntos[:,0]))]
            ax.plot(p_ordenados[:, 0], p_ordenados[:, 1], 'b-', lw=3)
    elif len(puntos) == 2:
        p_ordenados = puntos[np.lexsort((puntos[:,1], puntos[:,0]))]
        ax.plot(p_ordenados[:, 0], p_ordenados[:, 1], 'b-', lw=3)
    elif len(puntos) == 1:
        ax.plot(puntos[0][0], puntos[0][1], 'bo', markersize=8)
    
    if puntos_top: # si existen puntos en el límite superior, dibuja una flecha en el eje Y
        x_promedio = np.mean(puntos_top)
        ax.annotate('', xy=(x_promedio, max_y * 0.98),
                    xytext=(x_promedio, max_y * 0.72),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))
                    
    if puntos_right: # si existen puntos en el límite derecho, dibuja una flecha en el eje X
        y_promedio = np.mean(puntos_right)
        ax.annotate('', xy=(max_x * 0.98, y_promedio),
                    xytext=(max_x * 0.72, y_promedio),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))


# Grafica el sistema de inecuaciones, pinta el area factible y llama a la funcion de evaluacion     
def graficar_y_resolver(Matriz, Restriccion, intersecciones, obj, tipo):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Método Gráfico (Restricciones)")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    
    # Configurar Límites
    max_x = max([x for x, y in intersecciones] + [MIN_RANGO]) * ESCALA_EJE
    max_y = max([y for x, y in intersecciones] + [MIN_RANGO]) * ESCALA_EJE
    ax.set_xlim(-max_x * MARGEN_NEG, max_x)
    ax.set_ylim(-max_y * MARGEN_NEG, max_y)
    
    # 1. Graficar rectas
    x_vals = np.linspace(-max_x * MARGEN_NEG, max_x, N_PUNTOS_X)
    dibujar_lineas_restriccion(ax, Matriz, x_vals)
    
    # 2. Obtener puntos factibles
    optimos = obtener_puntos_factibles(intersecciones, Matriz, Restriccion)
    if not optimos:
        print("\nNo hay región factible.")
        plt.show()
        return
    
    # 3. Dibujar región factible (polígono y flechas)
    dibujar_poligono_factible(ax, optimos, Matriz, Restriccion, max_x, max_y)
    
    # 4. Marcar Vértices y Punto Óptimo
    if intersecciones:
        ix, iy = zip(*intersecciones)
        ax.plot(ix, iy, 'ko', markersize=4, zorder=5)
    
    evaluar_optimo(optimos, obj, tipo, ax)
    ax.legend(fontsize=8, loc='upper right')
    plt.show()


# funcion principal que coordina el flujo del programa
def iniciar_app():
    try:
        Matriz, Restriccion, obj, tipo_funcion = ingresar_datos()
        mostrar_ecuaciones(Matriz, Restriccion)
        intersecciones = calcular_intersecciones(Matriz)
        graficar_y_resolver(Matriz, Restriccion, intersecciones, obj, tipo_funcion)
    except KeyboardInterrupt:
        print("\n[!] Programa interrumpido por el usuario.")
    except Exception as e:
        print(f"\n[!] Ocurrió un error inesperado durante la ejecución: {e}")
        print("Asegúrese de cerrar las ventanas del gráfico antes de continuar o revise los datos ingresados.")

if __name__ == "__main__":
    iniciar_app()
