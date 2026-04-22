import numpy as np
import pandas as pd

def ingresar_datos():
    print("=== MÉTODO SIMPLEX DE 2 FASES ===")
    tipo_opt = input("¿Desea Maximizar o Minimizar? (max/min): ").strip().lower()
    
    n_vars = int(input("Ingrese la cantidad de variables de decisión: "))
    n_rest = int(input("Ingrese la cantidad de restricciones: "))
    
    print(f"\nIngrese los {n_vars} coeficientes de la Función Objetivo (separados por espacio):")
    # Acepta comas y puntos en la función objetivo
    entrada_c = input().replace(',', '.')
    c = np.array([float(x) for x in entrada_c.split()])
    
    if tipo_opt == 'max':
        c = -c # Convertimos a minimización internamente
        
    A = []
    b = []
    signos = []
    
    for i in range(n_rest):
        print(f"\n--- Restricción {i+1} ---")
        # Acepta comas y puntos en los coeficientes
        entrada_coefs = input(f"Coeficientes de las {n_vars} variables (separados por espacio): ").replace(',', '.')
        coefs = [float(x) for x in entrada_coefs.split()]
        
        signo = input("Signo (<=, >=, =): ").strip()
        
        # Acepta comas y puntos en el lado derecho
        entrada_ld = input("Lado derecho (término independiente): ").replace(',', '.')
        lado_derecho = float(entrada_ld)
        
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

def resolver_simplex(tablero, n_vars_totales, fase):
    filas, cols = tablero.shape
    
    while True:
        costos_reducidos = tablero[0, :-1]
        if np.all(costos_reducidos >= -1e-7):
            break
            
        col_pivote = np.argmin(costos_reducidos)
        
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
    
    tablero = np.zeros((n_rest + 1, n_totales + 1))
    tablero[1:, :n_vars] = A
    tablero[1:, -1] = b
    
    # Generar nombres de columnas dinámicamente
    cols_vars = [f"X{i+1}" for i in range(n_vars)]
    cols_holg_exc = []
    cols_artif = []
    
    idx_holgura_exceso = n_vars
    idx_artificial = n_vars + n_holgura + n_exceso
    
    filas_artificiales = []
    
    c_s, c_e, c_a = 1, 1, 1 # Contadores para nombres (S1, E1, A1...)
    
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

    columnas_fase1 = cols_vars + cols_holg_exc + cols_artif + ["LD"]
    columnas_fase2 = cols_vars + cols_holg_exc + ["LD"]

    # FASE 1
    if n_artificiales > 0:
        for i in range(n_vars + n_holgura + n_exceso, n_totales):
            tablero[0, i] = 1
            
        for fila in filas_artificiales:
            tablero[0, :] -= tablero[fila, :]
            
        mostrar_tablero(tablero, columnas_fase1, "TABLERO INICIAL (INICIO DE FASE 1)")
        
        tablero = resolver_simplex(tablero, n_totales, 1)
        
        if tablero is None:
            return
            
        if abs(tablero[0, -1]) > 1e-7:
            print("\nEl problema es INFACTIBLE (no tiene solución posible).")
            return
    else:
        print("\nNo se requieren variables artificiales. Omitiendo Fase 1.")

    # FASE 2
    if n_artificiales > 0:
        cols_a_mantener = list(range(n_vars + n_holgura + n_exceso)) + [-1]
        tablero = tablero[:, cols_a_mantener]
        
    tablero[0, :] = 0
    tablero[0, :n_vars] = c
    
    filas, cols = tablero.shape
    for i in range(1, filas):
        for j in range(cols - 1):
            if tablero[i, j] == 1 and np.sum(np.abs(tablero[1:, j])) == 1:
                if tablero[0, j] != 0:
                    tablero[0, :] -= tablero[0, j] * tablero[i, :]
                break
                
    mostrar_tablero(tablero, columnas_fase2, "TABLERO INICIAL FASE 2 (Variables artificiales eliminadas)")
                
    tablero = resolver_simplex(tablero, n_vars + n_holgura + n_exceso, 2)
    
    if tablero is not None:
        mostrar_tablero(tablero, columnas_fase2, "TABLERO ÓPTIMO FINAL")
        
        print("\n=== SOLUCIÓN ÓPTIMA ===")
        z_opt = -tablero[0, -1] if tipo_opt == 'max' else tablero[0, -1]
        print(f"Valor Óptimo (Z) = {np.round(z_opt, 4)}")
        
        for j in range(n_vars):
            valor = 0
            if np.sum(np.abs(tablero[1:, j])) == 1 and np.max(tablero[1:, j]) == 1:
                fila_basica = np.where(tablero[1:, j] == 1)[0][0] + 1
                valor = tablero[fila_basica, -1]
            print(f"X{j+1} = {np.round(valor, 4)}")

if __name__ == "__main__":
    metodo_dos_fases()