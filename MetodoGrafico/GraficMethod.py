# Metodo Grafico
# Algoritmo de Optimizacion
# by: Alex, Matias
# UCT
# dos fases

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from itertools import combinations # esta libreria permite calcular la regla de Cramer para todos los cruces en un solo bloque compacto.

#Elimina los decimales si el número es entero, de lo contrario lo formatea a 2 decimales.

def formato_numero(num):
    return f"{int(num)}" if num == int(num) else f"{num:.2f}"

# Solicita al usuario de forma dinámica las restricciones y la función objetivo.
def ingresar_datos():
    n = int(input("\nNúmero de restricciones: "))
    M, T = [], []
    for i in range(n):
        print(f"\nIngrese los datos para la restricción {i+1}:")
        a = float(input("Coeficiente de x (a): "))
        b = float(input("Coeficiente de y (b): "))
        
        t = input("Tipo de restricción (<=, >=, =): ").strip()
        while t not in ['<=', '>=', '=']: 
            t = input("Tipo inválido. Ingrese <=, >= o =: ").strip()
            
        c = float(input("Término constante (c): "))
        M.append([a, b, c])
        T.append(t)
    
    print("\nIngrese los coeficientes de la función objetivo Z:")
    c_x = float(input("Coeficiente de x: "))
    c_y = float(input("Coeficiente de y: "))
    
    print("\n¿Qué desea hacer con la función objetivo?\n1. Maximizar\n2. Minimizar")
    tipo_opt = 'max' if input("Seleccione una opción (1/2): ").strip() == '1' else 'min'
    
    return np.array(M), T, (c_x, c_y), tipo_opt

# Muestra el sistema de restricciones ingresado por el usuario en formato algebraico.
def mostrar_ecuaciones(M, T):
    print("\n--- Restricciones ---")
    for i, (a, b, c) in enumerate(M):
        signo = "+" if b >= 0 else "-"
        print(f"Restricción {i+1}: {formato_numero(a)}x {signo} {formato_numero(abs(b))}y {T[i]} {formato_numero(c)}")
    print("Condición de no negatividad: x >= 0, y >= 0")


# Calcula las intersecciones entre todas las rectas y los ejes usando la regla de Cramer, combina todos los pares posibles de ecuaciones para encontrar sus puntos de cruce.
def calcular_intersecciones(M):
    print("\n--- Puntos de Intersección ---")
    intersecciones = []
    
    # Se añaden los ejes x=0 (0x + 1y = 0) e y=0 (1x + 0y = 0) como rectas adicionales al sistema
    ejes = np.array([[0, 1, 0], [1, 0, 0]])
    todas_rectas = np.vstack((M, ejes))
    
    # Evaluar la intersección de cada par posible de rectas usando itertools.combinations
    for r1, r2 in combinations(todas_rectas, 2):
        det = r1[0] * r2[1] - r2[0] * r1[1]
        if det != 0: # Si el determinante no es 0, no son rectas paralelas
            x = (r1[2] * r2[1] - r2[2] * r1[1]) / det
            y = (r1[0] * r2[2] - r2[0] * r1[2]) / det
            
            # Considerar solo intersecciones en el primer cuadrante (x>=0, y>=0)
            if x >= -1e-5 and y >= -1e-5:
                intersecciones.append((abs(x), abs(y))) # abs() evita mostrar -0.0
                
    # Filtrar puntos duplicados con tolerancia a decimales (redondeando)
    inter_unicas = list(set((round(x, 4), round(y, 4)) for x, y in intersecciones))
    for x, y in inter_unicas:
        print(f"Intersección: ({formato_numero(x)}, {formato_numero(y)})")
        
    return inter_unicas

# Evalúa la función objetivo en los vértices de la región factible para hallar la solución óptima
def evaluar_optimo(optimos, obj, tipo_opt, ax):
    print("\n--- Evaluación de Z ---")
    c_x, c_y = obj
    mejor_z, mejor_pt = None, None
    
    for x, y in optimos:
        z = c_x * x + c_y * y
        print(f"Z({formato_numero(x)}, {formato_numero(y)}) = {formato_numero(z)}")
        
        # Guardar el mejor valor según el objetivo solicitado (maximizar o minimizar)
        if mejor_z is None or (tipo_opt == 'max' and z > mejor_z) or (tipo_opt == 'min' and z < mejor_z):
            mejor_z = z
            mejor_pt = (x, y)
            
    if mejor_pt:
        x_o, y_o = mejor_pt
        tipo_texto = "Máximo" if tipo_opt == 'max' else "Mínimo"
        print(f"\nPunto óptimo ({tipo_texto}): ({formato_numero(x_o)}, {formato_numero(y_o)}) con Z = {formato_numero(mejor_z)}")
        
        # Graficar y resaltar el punto óptimo en el plano
        ax.plot(x_o, y_o, 'ro', markersize=10, label=f'Solución Óptima ({tipo_texto})', zorder=10)
        ax.annotate(f"SOLUCIÓN\n({formato_numero(x_o)}, {formato_numero(y_o)})\nZ={formato_numero(mejor_z)}",
                    (x_o, y_o), xytext=(15, 15), textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5), fontweight='bold')

# Grafica el sistema de inecuaciones, pinta el área factible y llama a la función de evaluación
def graficar_y_resolver(M, T, intersecciones, obj, tipo_opt):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Método Gráfico (Restricciones)")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    
    # Calcular límites del gráfico dinámicamente en base a las intersecciones encontradas
    max_x = max([x for x, y in intersecciones] + [10]) * 1.1
    max_y = max([y for x, y in intersecciones] + [10]) * 1.1
    ax.set_xlim(-max_x*0.05, max_x)
    ax.set_ylim(-max_y*0.05, max_y)
    x_vals = np.linspace(-max_x*0.05, max_x, 400)
    
    # Obtener paleta de colores dinámica de matplotlib (elimina el hardcoding de colores)
    colores = plt.cm.get_cmap('tab10').colors
    
    plt.ion() # Activa el modo interactivo para animar la aparición de rectas
    
    # 1. Graficar cada recta de forma incremental
    for i, (a, b, c) in enumerate(M):
        color = colores[i % len(colores)]
        if b != 0:
            y_vals = (c - a * x_vals) / b
            ax.plot(x_vals, y_vals, color=color, label=f"R{i+1}: {formato_numero(a)}x + {formato_numero(b)}y = {formato_numero(c)}")
        else:
            ax.axvline(x=c/a, color=color, label=f"R{i+1}: x = {formato_numero(c/a)}")
            
        ax.legend(fontsize=8, loc='upper right')
        plt.pause(0.8) # Pausa para visualizar la animación
        
    plt.ioff() # Desactiva el modo interactivo
            
    # 2. Filtrar vértices que cumplan con TODAS las restricciones (Región factible)
    optimos = []
    for x, y in intersecciones: 
        factible = True
        for (a, b, c), t in zip(M, T):
            valor = a * x + b * y
            if (t == '<=' and valor > c + 1e-5) or (t == '>=' and valor < c - 1e-5) or (t == '=' and abs(valor - c) > 1e-5):
                factible = False
                break
        if factible:
            optimos.append((x, y))
            
    # Filtrar duplicados en los óptimos (por tolerancia de decimales)
    optimos = list(set((round(x, 4), round(y, 4)) for x, y in optimos))
            
    if not optimos:
        print("\nNo hay vértices para definir una región factible (Infactible).")
        plt.show()
        return

    # 3. Pintar región factible utilizando ConvexHull para polígonos
    if len(optimos) > 2:
        try:
            puntos = np.array(optimos)
            hull = ConvexHull(puntos)
            pts_ord = puntos[hull.vertices]
            ax.fill(pts_ord[:, 0], pts_ord[:, 1], alpha=0.3, color='skyblue', label='Región factible')
            ax.plot(np.append(pts_ord[:,0], pts_ord[0,0]), np.append(pts_ord[:,1], pts_ord[0,1]), 'blue', lw=1)
        except Exception:
            # Captura casos donde ConvexHull falla (p.ej. puntos colineales)
            ax.plot([p[0] for p in optimos], [p[1] for p in optimos], 'b-', lw=3, label='Región factible')
    elif len(optimos) == 2:
        ax.plot([p[0] for p in optimos], [p[1] for p in optimos], 'b-', lw=3, label='Región factible (Línea)')
    elif len(optimos) == 1:
        ax.plot(optimos[0][0], optimos[0][1], 'bo', markersize=8, label='Región factible (Punto)')

    # Graficar todos los puntos de intersección
    for x, y in intersecciones:
        ax.plot(x, y, 'ko', markersize=4, zorder=5)

    # 4. Evaluar la función objetivo y destacar la solución
    evaluar_optimo(optimos, obj, tipo_opt, ax)

    ax.legend(fontsize=8, loc='upper right')
    plt.draw()
    plt.show()

# Función principal que coordina el flujo completo del algoritmo.
def iniciar_app():
    M, T, obj, tipo_opt = ingresar_datos()
    mostrar_ecuaciones(M, T)
    intersecciones = calcular_intersecciones(M)
    graficar_y_resolver(M, T, intersecciones, obj, tipo_opt)

if __name__ == "__main__":
    iniciar_app()
