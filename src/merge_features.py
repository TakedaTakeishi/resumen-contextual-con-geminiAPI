import json
import os
import glob
import re

def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error leyendo {filepath}: {e}")
        return []

def save_json(data, filepath):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Guardado exitosamente en: {filepath}")
    except Exception as e:
        print(f"Error guardando {filepath}: {e}")

def get_start_hour(timeframe):
    # Extraer la hora de inicio del timeframe "HH:MM-HH:MM"
    match = re.match(r"(\d{2}):(\d{2})", timeframe)
    if match:
        return int(match.group(1)) * 60 + int(match.group(2)) # Minutos desde inicio del día
    return 0

def main():
    # Rutas base
    base_dir = r"c:\Users\Joni\Documents\Universidad\6to_Semestre\Hackaton\IPN\emmbeddings"
    processed_dir = os.path.join(base_dir, "data", "processed")
    raw_extras_dir = os.path.join(base_dir, "data", "raw", "features extra")
    
    # Archivo original
    original_features_path = os.path.join(processed_dir, "features.json")
    
    # Carpeta de destino y archivo de destino
    output_dir = os.path.join(processed_dir, "features")
    output_file = os.path.join(output_dir, "features.json")
    
    all_data = []
    
    # 1. Leer features.json original
    if os.path.exists(original_features_path):
        print(f"Leyendo archivo base: {original_features_path}")
        data = load_json(original_features_path)
        if isinstance(data, list):
            all_data.extend(data)
            print(f"  - Agregados {len(data)} registros.")
        else:
            print("  - El archivo base no es una lista JSON válida.")
    else:
        print(f"No se encontró el archivo base: {original_features_path}")

    # 2. Leer archivos extra
    if os.path.exists(raw_extras_dir):
        extra_files = glob.glob(os.path.join(raw_extras_dir, "*.json"))
        print(f"Buscando extras en: {raw_extras_dir}")
        for extra_file in extra_files:
            print(f"Leyendo extra: {extra_file}")
            data = load_json(extra_file)
            if isinstance(data, list):
                all_data.extend(data)
                print(f"  - Agregados {len(data)} registros.")
            else:
                print(f"  - {extra_file} no es una lista JSON válida.")
    else:
        print(f"No existe el directorio de extras: {raw_extras_dir}")

    # 3. Ordenar y corregir IDs
    print("Ordenando datos por horario y reasignando IDs...")
    
    # Ordenar por timeframe
    all_data.sort(key=lambda x: get_start_hour(x.get('batch_meta', {}).get('timeframe', '00:00')))
    
    # Reasignar batch_id secuencialmente basado en cambios de timeframe
    current_batch_id = 0
    last_timeframe = None
    
    for item in all_data:
        timeframe = item.get('batch_meta', {}).get('timeframe')
        
        if timeframe != last_timeframe:
            current_batch_id += 1
            last_timeframe = timeframe
            
        item['batch_meta']['batch_id'] = current_batch_id

    # 4. Guardar resultado
    print(f"Total de registros combinados y corregidos: {len(all_data)}")
    save_json(all_data, output_file)

if __name__ == "__main__":
    main()
