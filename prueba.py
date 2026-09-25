import pandas as pd
import numpy as np
import os


#1 lectura y auditoria de los datos---------------------------------------------------------------------------------------------------------------------

#debug por si no se encuentra el archivo de excel (agarra el primer archivo encontrado con la extension)
files = os.listdir('.')
print("Archivos en directorio:", files)
excel_file = [f for f in files if f.endswith('.xlsx') or f.endswith('.xls')][0]
print("Archivo Excel seleccionado:", excel_file)

xls = pd.ExcelFile(excel_file)
print("Hojas encontradas:", xls.sheet_names)
df = pd.read_excel(excel_file, sheet_name=0)
print("Shape original:", df.shape)

#informacion en general del dataset
df_info = pd.DataFrame({
    'Column': df.columns,
    'DataType': df.dtypes.values,
    'Non-Null Count': df.notnull().sum().values,
    'Missing Count': df.isnull().sum().values,
    'Missing %': (df.isnull().sum().values / len(df) * 100).round(2),
    'Min': [df[col].min() if pd.api.types.is_numeric_dtype(df[col]) and df[col].notnull().any() else 'N/A' for col in df.columns],
    'Max': [df[col].max() if pd.api.types.is_numeric_dtype(df[col]) and df[col].notnull().any() else 'N/A' for col in df.columns],
    'Mean': [round(df[col].mean(), 2) if pd.api.types.is_numeric_dtype(df[col]) and df[col].notnull().any() else 'N/A' for col in df.columns]
})
print("\n=== RESUMEN DE AUDITORÍA DE DATOS ===")
print(df_info.to_string())



#2 Nowcast---------------------------------------------------------------------------------------------------------------------

def NowCast(valores, PM):
    """
    Calcula el NowCast según NOM-172-SEMARNAT-2023.
    Args:
        valores: Lista (longitud <=12) de concentraciones de PM (viejo -> reciente)
        PM: 0 para PM10, 1 para PM2.5.
    Returns:
        Entero con el promedio ponderado NowCast (µg/m³) o np.nan si no cumple.
    """
    # Condición a): Al menos 2 de las 3 horas más recientes deben existir
    ultimas_3 = valores[-3:] if len(valores) >= 3 else valores
    if sum(x is not None and not pd.isna(x) for x in ultimas_3) < 2:
        return np.nan

    # Construir lista (valor, hora_consecutiva) respetando huecos
    valores_rev = valores[::-1]
    datos = []
    i = 0
    for v in valores_rev:
        if v is not None and not pd.isna(v):
            datos.append((float(v), i))
        i += 1

    if len(datos) < 2:
        return np.nan

    # Rango y factor W (>=0.5)
    solo_val = [v for v, _ in datos]
    max_v = max(solo_val)
    rango = max_v - min(solo_val)
    
    w_raw = round(1 - (rango / max_v), 2) if max_v > 0 else 0.5
    W = w_raw if w_raw >= 0.5 else 0.5

    # Promedio ponderado
    num = 0.0
    den = 0.0
    for v, hora_consec in datos:
        peso = (W ** hora_consec)
        num += v * peso
        den += peso

    if den == 0:
        return np.nan

    promedio = round(num / den, 0)

    # Ajuste por tipo de PM (NOM 172, Anexo A): 0.714 para PM10, 0.694 para PM2.5
    if PM == 0: # PM10
        promedio = round(promedio * 0.714, 0)
    else: # PM2.5
        promedio = round(promedio * 0.694, 0)

    return float(promedio)

def calcular_nowcast_serie(serie, pm_flag):
    """Aplica la función NowCast a lo largo de una Serie temporal de pandas"""
    vals = serie.tolist()
    resultados = []
    for i in range(len(vals)):
        # Tomar ventana móvil de hasta 12 horas hacia atrás
        win = vals[max(0, i - 11): i + 1]
        if len(win) < 12:
            win = [None] * (12 - len(win)) + win
        resultados.append(NowCast(win, pm_flag))
    return resultados



#3 procesado y calculos---------------------------------------------------------------------------------------------------------------------
#estampilla de tiempo
df['TIMESTAMP'] = pd.to_datetime(df['DATE']).dt.normalize() + pd.to_timedelta(df['HOUR'], unit='h')
df = df.sort_values(by='TIMESTAMP').reset_index(drop=True)
#conversión de O3 (ppm a ppb)
df['O3_ppb'] = df['O3'] * 1000
#limpieza de lecturas negativas
for col in ['O3_ppb', 'PM10', 'PM2.5']:
    df.loc[df[col] < 0, col] = np.nan
#promedios según la norma
# Ozono: Promedio móvil de 8h (suficiencia >= 75% -> min_periods=6)
df['O3_8h'] = df['O3_ppb'].rolling(window=8, min_periods=6).mean()
# Partículas: NowCast
df['PM10_NowCast'] = calcular_nowcast_serie(df['PM10'], pm_flag=0)
df['PM25_NowCast'] = calcular_nowcast_serie(df['PM2.5'], pm_flag=1)

#4 categorizacion y calidad del aire---------------------------------------------------------------------------------------------------------------------

def cat_o3(v):
    if pd.isna(v): return np.nan
    return 'Buena' if v <= 51 else 'Aceptable' if v <= 70 else 'Mala' if v <= 92 else 'Muy Mala' if v <= 114 else 'Extremadamente Mala'

def cat_pm10(v):
    if pd.isna(v): return np.nan
    return 'Buena' if v <= 50 else 'Aceptable' if v <= 75 else 'Mala' if v <= 155 else 'Muy Mala' if v <= 235 else 'Extremadamente Mala'

def cat_pm25(v):
    if pd.isna(v): return np.nan
    return 'Buena' if v <= 25 else 'Aceptable' if v <= 45 else 'Mala' if v <= 79 else 'Muy Mala' if v <= 147 else 'Extremadamente Mala'

df['Cat_O3'] = df['O3_8h'].apply(cat_o3)
df['Cat_PM10'] = df['PM10_NowCast'].apply(cat_pm10)
df['Cat_PM25'] = df['PM25_NowCast'].apply(cat_pm25)

#nivel global de riesgo y contaminante dominante
cat_order = {'Buena': 1, 'Aceptable': 2, 'Mala': 3, 'Muy Mala': 4, 'Extremadamente Mala': 5}
inv_cat = {1: 'Buena', 2: 'Aceptable', 3: 'Mala', 4: 'Muy Mala', 5: 'Extremadamente Mala'}

def get_global_info(row):
    cats = [('O3', row['Cat_O3']), ('PM10', row['Cat_PM10']), ('PM25', row['Cat_PM25'])]
    validos = [(p, c, cat_order[c]) for p, c in cats if pd.notna(c) and c in cat_order]
    
    if not validos:
        return np.nan, np.nan
    
    max_riesgo = max([item[2] for item in validos])
    categoria_global = inv_cat[max_riesgo]
    dominantes = [p for p, c, r in validos if r == max_riesgo]
    
    return categoria_global, ", ".join(dominantes)

res = df.apply(get_global_info, axis=1)
df['Categoria_Global'] = [r[0] for r in res]
df['Contaminante_Dominante'] = [r[1] for r in res]

#resultados
archivo_salida = 'LasPintasBDprocesada.csv'
df.to_csv(archivo_salida, index=False)

print(f"\nProceso completado. Archivo generado: '{archivo_salida}'")
print(f"Registros procesados: {len(df)} | Rango: {df['TIMESTAMP'].min()} a {df['TIMESTAMP'].max()}")