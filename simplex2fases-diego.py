import numpy as np
import pandas as pd

def leer_entero(mensaje):
    """Lee un número entero y asegura que sea mayor a 0."""
    while True:
        try:
            valor = int(input(mensaje))
            if valor <= 0:
                print("Error: Ingrese un número entero positivo mayor a 0.")
                continue
            return valor
        except ValueError:
            print("Error: Entrada inválida. Debe ingresar un número entero.")

def leer_flotante(mensaje):
    """Lee un número decimal, aceptando puntos o comas."""
    while True:
        try:
            entrada = input(mensaje).replace(',', '.')
            return float(entrada)
        except ValueError:
            print("Error: Entrada inválida. Debe ingresar un número válido.")

def leer_lista_flotantes(mensaje, n_esperado):
    """Lee varios números en una sola línea separados por espacio y valida la cantidad."""
    while True:
        try:
            entrada = input(mensaje).replace(',', '.')
            valores = [float(x) for x in entrada.split()]
            if len(valores) == n_esperado:
                return valores
            else:
                print(f"Error: Se esperaban {n_esperado} valores, pero ingresaste {len(valores)}.")
        except ValueError:
            print("Error: Asegúrate de ingresar solo números separados por un espacio.")

def leer_signo(mensaje):
    """Valida que el signo ingresado sea correcto para Simplex."""
    while True:
        signo = input(mensaje).strip()
        if signo in ['<=', '>=', '=']:
            return signo
        print("Error: Ingresa un signo válido ('<=', '>=', o '=').")

def leer_tipo_optimizacion(mensaje):
    """Valida la opción de maximizar o minimizar."""
    while True:
        tipo = input(mensaje).strip().lower()
        if tipo in ['max', 'min']:
            return tipo
        print("Error: Por favor, escribe solo 'max' o 'min'.")


# --- FUNCIÓN PRINCIPAL DE ENTRADA DE DATOS ---

def ingresar_datos():
    print("=== MÉTODO SIMPLEX DE 2 FASES ===")
    
    tipo_opt = leer_tipo_optimizacion("¿Desea Maximizar o Minimizar? (max/min): ")
    n_vars = leer_entero("Ingrese la cantidad de variables de decisión: ")
    n_rest = leer_entero("Ingrese la cantidad de restricciones: ")
    
    print("\n--- Función Objetivo ---")
    c_list = leer_lista_flotantes(f"Ingrese los {n_vars} coeficientes (separados por espacio): ", n_vars)
    c = np.array(c_list)
    
    if tipo_opt == 'max':
        c = -c # Convertimos a minimización internamente
        
    A = []
    b = []
    signos = []
    
    for i in range(n_rest):
        print(f"\n--- Restricción {i+1} ---")
        coefs = leer_lista_flotantes(f"Coeficientes de las {n_vars} variables (separados por espacio): ", n_vars)
        signo = leer_signo("Signo (<=, >=, =): ")
        lado_derecho = leer_flotante("Lado derecho (término independiente): ")
        
        if lado_derecho < 0:
            coefs = [-x for x in coefs]
            lado_derecho = -lado_derecho
            if signo == '<=': signo = '>='
            elif signo == '>=': signo = '<='
            
        A.append(coefs)
        signos.append(signo)
        b.append(lado_derecho)
        
    return c, np.array(A), np.array(b), signos, n_vars, n_rest, tipo_opt

def mostrar_tablero(tablero, columnas, titulo):
    """Muestra el tablero Simplex usando Pandas para formato de tabla"""
    df = pd.DataFrame(tablero, columns=columnas).round(3)
    filas = ['Z'] + [f'Fila {i}' for i in range(1, len(tablero))]
    df.index = filas
    print(f"\n{titulo}")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70)

def resolver_simplex(tablero, fase):
    filas, cols = tablero.shape
    
    while True:
        # 1. Condición de parada y búsqueda de pivote
        # Buscamos en los costos reducidos (Fila 0), PERO ignoramos la columna 0 (que es Z) 
        # y la última columna (que es el lado derecho).
        costos_reducidos = tablero[0, 1:-1]
        
        if np.all(costos_reducidos >= -1e-7):
            break
            
        # +1 porque recortamos la columna Z al buscar, así recuperamos el índice real
        col_pivote = np.argmin(costos_reducidos) + 1 
        
        # 3. Variable de salida (prueba del cociente mínimo)
        columna_entrada = tablero[1:, col_pivote]
        lados_derechos = tablero[1:, -1]
        
        cocientes = np.full(filas - 1, np.inf)
        for i in range(filas - 1):
            if columna_entrada[i] > 1e-7:
                cocientes[i] = lados_derechos[i] / columna_entrada[i]
                
        if np.all(cocientes == np.inf):
            print("\nEl problema tiene solución ilimitada.")
            return None
            
        fila_pivote = np.argmin(cocientes) + 1 
        
        # 4. Operaciones elementales de fila (Gauss-Jordan)
        elemento_pivote = tablero[fila_pivote, col_pivote]
        tablero[fila_pivote, :] = tablero[fila_pivote, :] / elemento_pivote
        
        for i in range(filas):
            if i != fila_pivote:
                factor = tablero[i, col_pivote]
                tablero[i, :] -= factor * tablero[fila_pivote, :]
                
    return tablero

def metodo_dos_fases():
    c, A, b, signos, n_vars, n_rest, tipo_opt = ingresar_datos()
    
    n_holgura = signos.count('<=')
    n_exceso = signos.count('>=')
    n_artificiales = signos.count('>=') + signos.count('=')
    
    n_totales = n_vars + n_holgura + n_exceso + n_artificiales
    
    # +2 en columnas: 1 para la columna de Z al inicio, y 1 para el LD al final
    tablero = np.zeros((n_rest + 1, n_totales + 2))
    
    # LA COLUMNA Z: 1 en la función objetivo, 0 en las restricciones
    tablero[0, 0] = 1 
    
    # Las variables originales se desplazan +1 por la columna Z
    tablero[1:, 1:n_vars+1] = A
    tablero[1:, -1] = b
    
    cols_vars = [f"X{i+1}" for i in range(n_vars)]
    cols_holg_exc = []
    cols_artif = []
    
    idx_holgura_exceso = n_vars + 1 # Se desplaza 1 por la Z
    idx_artificial = n_vars + n_holgura + n_exceso + 1 # Se desplaza 1 por la Z
    
    filas_artificiales = []
    c_s, c_e, c_a = 1, 1, 1 
    
    for i, signo in enumerate(signos):
        if signo == '<=':
            tablero[i+1, idx_holgura_exceso] = 1
            idx_holgura_exceso += 1
            cols_holg_exc.append(f"S{c_s}")
            c_s += 1
        elif signo == '>=':
            tablero[i+1, idx_holgura_exceso] = -1
            idx_holgura_exceso += 1
            cols_holg_exc.append(f"E{c_e}")
            c_e += 1
            
            tablero[i+1, idx_artificial] = 1
            filas_artificiales.append(i+1)
            idx_artificial += 1
            cols_artif.append(f"A{c_a}")
            c_a += 1
        elif signo == '=':
            tablero[i+1, idx_artificial] = 1
            filas_artificiales.append(i+1)
            idx_artificial += 1
            cols_artif.append(f"A{c_a}")
            c_a += 1

    columnas_fase1 = ["Z"] + cols_vars + cols_holg_exc + cols_artif + ["LD"]
    columnas_fase2 = ["Z"] + cols_vars + cols_holg_exc + ["LD"]

    # --- FASE 1 ---
    if n_artificiales > 0:
        # Llenar 1s en la fila 0 solo para las columnas artificiales
        for i in range(n_vars + n_holgura + n_exceso + 1, n_totales + 1):
            tablero[0, i] = 1
            
        for fila in filas_artificiales:
            tablero[0, :] -= tablero[fila, :]
            
        mostrar_tablero(tablero, columnas_fase1, "TABLERO INICIAL (INICIO DE FASE 1)")
        tablero = resolver_simplex(tablero, 1)
        
        if tablero is None: return
            
        if abs(tablero[0, -1]) > 1e-7:
            print("\nEl problema es INFACTIBLE (no tiene solución posible).")
            return
    else:
        print("\nNo se requieren variables artificiales. Omitiendo Fase 1.")

    # --- FASE 2 ---
    if n_artificiales > 0:
        # Mantenemos Z(0), las vars de decisión, holguras, excesos, y el LD(-1)
        cols_a_mantener = [0] + list(range(1, n_vars + n_holgura + n_exceso + 1)) + [-1]
        tablero = tablero[:, cols_a_mantener]
        
    # Resetear F.O. pero manteniendo Z en 1
    tablero[0, :] = 0
    tablero[0, 0] = 1
    tablero[0, 1:n_vars+1] = c # Insertamos la F.O original desplazada en 1
    
    # Restaurar la estructura de la base
    filas, cols = tablero.shape
    for i in range(1, filas):
        # Buscamos base ignorando la columna Z (j=0) y el LD (j=cols-1)
        for j in range(1, cols - 1):
            if tablero[i, j] == 1 and np.sum(np.abs(tablero[1:, j])) == 1:
                if tablero[0, j] != 0:
                    tablero[0, :] -= tablero[0, j] * tablero[i, :]
                break
                
    mostrar_tablero(tablero, columnas_fase2, "TABLERO INICIAL FASE 2")
    tablero = resolver_simplex(tablero, 2)
    
    if tablero is not None:
        mostrar_tablero(tablero, columnas_fase2, "TABLERO ÓPTIMO FINAL")
        
        print("\n=== SOLUCIÓN ÓPTIMA ===")
        z_opt = -tablero[0, -1] if tipo_opt == 'max' else tablero[0, -1]
        print(f"Valor Óptimo (Z) = {np.round(z_opt, 4)}")
        
        for j in range(1, n_vars + 1):
            valor = 0
            if np.sum(np.abs(tablero[1:, j])) == 1 and np.max(tablero[1:, j]) == 1:
                fila_basica = np.where(tablero[1:, j] == 1)[0][0] + 1
                valor = tablero[fila_basica, -1]
            print(f"X{j} = {np.round(valor, 4)}")
if __name__ == "__main__":
    metodo_dos_fases()