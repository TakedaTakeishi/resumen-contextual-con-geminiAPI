import json
import os
import re
import statistics
import time
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'DATOS_CRUDOS_L1_COMPLETO.json')
REPORTS_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'INFORMES_SIEMONS_DATOS_CRUDOS_L1_COMPLETOS.json')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'features.json')

# Configuration
BATCH_SIZE = 200

# Set API Key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")

genai.configure(api_key=api_key)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_numerical_features(batch_data, station_name):
    """
    Extracts and averages numerical features for a specific station from the batch.
    """
    temps = []
    humidities = []
    precips = []
    jam_levels = []

    # Normalize station name for matching
    station_clean = station_name.lower()

    for item in batch_data:
        # Weather data (assuming it applies to the general area if no specific station match, 
        # or we could try to map 'station_id' to Metro stations)
        # For this hackathon scope, we'll take all weather in the batch as context for the line segment
        if 'temp' in item and item.get('temp') is not None:
            temps.append(item['temp'])
        if 'humidity' in item and item.get('humidity') is not None:
            humidities.append(item['humidity'])
        if 'precip_mm' in item and item.get('precip_mm') is not None:
            precips.append(item['precip_mm'])
        
        # Traffic data - check if location matches station
        if 'jam_level' in item and 'location' in item:
            loc = item['location'].lower()
            # Simple heuristic: if station name is in traffic location description
            if station_clean in loc:
                jam_levels.append(item['jam_level'])

    return {
        "temp": round(statistics.mean(temps), 1) if temps else None,
        "humidity": round(statistics.mean(humidities), 1) if humidities else None,
        "precip_mm": round(statistics.mean(precips), 1) if precips else None,
        "traffic_jam_level": round(statistics.mean(jam_levels), 1) if jam_levels else None
    }

def get_gemini_extraction(batch_data):
    """
    Uses Gemini API to:
    1. Identify stations mentioned in text (Tweets, Traffic, etc.)
    2. Generate a summary for each.
    """
    # Try using gemini-1.5-flash which is generally available
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # Construct prompt
    prompt = f"""
    Analiza el siguiente lote de datos crudos del Metro CDMX (Tweets, Clima, Tráfico).
    
    TAREA:
    1. Identifica las estaciones de Metro mencionadas explícitamente o por geolocalización.
    2. Para CADA estación detectada, genera un objeto JSON.
    
    FORMATO DE RESPUESTA (Array JSON):
    [
        {{
            "station_id": "Nombre de la estación (ej: Cuauhtémoc)",
            "resumen_texto": "Texto denso (máx 200 palabras). Combina QUEJAS + CONTEXTO OPERATIVO (clima, tráfico, causas). Redacción natural, SIN saludos, SIN listas."
        }}
    ]
    
    Ejemplo de resumen: "Usuarios reportan saturación severa y asfixia. Coincide con lluvia moderada (3mm) y alerta de tráfico nivel 5 en superficie. Se detectan menciones de humo."
    
    DATOS DEL LOTE:
    {json.dumps(batch_data, ensure_ascii=False)}
    """
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Robust JSON extraction: find the first '[' and last ']'
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx+1]
            return json.loads(json_str)
        else:
            # If no array found, maybe it returned a single object?
            if text.startswith('{') and text.endswith('}'):
                 return [json.loads(text)]
            
            print(f"Warning: No JSON found in response. Start: {text[:50]}...")
            return []
            
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return []

def process_batches():
    raw_data = load_json(RAW_DATA_PATH)
    reports_data = load_json(REPORTS_PATH)
    
    reports_map = {r['batch_id']: r for r in reports_data['reports']}
    
    final_output = []
    
    total_records = len(raw_data)
    num_batches = (total_records + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"Processing {total_records} records in {num_batches} batches using Gemini API...")

    for i in range(num_batches):
        batch_id = i + 1
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, total_records)
        
        batch_data = raw_data[start_idx:end_idx]
        
        print(f"Processing Batch {batch_id}...")
        
        # 1. Gemini Extraction
        extracted_stations = get_gemini_extraction(batch_data)
        
        # Sleep briefly to avoid hitting rate limits too hard
        time.sleep(1)
        
        # Get report for this batch to find the "Ground Truth" station if any
        report = reports_map.get(batch_id)
        timeframe = report.get('batch_timeframe', 'Unknown') if report else 'Unknown'
        
        # If the report mentions a specific station where a failure occurred, 
        # we MUST ensure it's in our extracted list (even if our simple mock missed it, 
        # though the mock should catch it if it's in the text).
        # For the purpose of the dataset generation, we iterate over what we "found".
        
        # Also, if the report says "falla_detectada": true, we want to make sure we align 
        # the Y_labels correctly for the station where it happened.
        
        for station_info in extracted_stations:
            station_name = station_info['station_id']
            
            # 2. Numerical Features Extraction
            num_features = extract_numerical_features(batch_data, station_name)
            
            # 3. Construct Y_labels
            y_labels = {
                "falla_detectada": False,
                "tipo_falla": None,
                "modo_falla_codigo": None,
                "impacto_operativo": "Normal",
                "retraso_minutos": 0,
                "descripcion_tecnica": None,
                "severidad": 0
            }
            
            if report and report.get('falla_detectada'):
                # Check if this station matches the one in the report
                report_location = report.get('location_segment')
                if report_location and report_location.lower() in station_name.lower():
                    y_labels["falla_detectada"] = True
                    y_labels["tipo_falla"] = report.get('root_cause_category')
                    y_labels["modo_falla_codigo"] = report.get('failure_mode_code')
                    y_labels["impacto_operativo"] = report.get('operational_impact')
                    y_labels["retraso_minutos"] = report.get('delay_minutes')
                    y_labels["descripcion_tecnica"] = report.get('incident_description')
                    
                    # Map severity
                    impact = report.get('operational_impact', '')
                    if 'Suspensión' in impact:
                        y_labels["severidad"] = 2
                    elif 'Retraso' in impact:
                        y_labels["severidad"] = 1
                    else:
                        y_labels["severidad"] = 0

            # 4. Assemble Final Object
            entry = {
                "batch_meta": {
                    "batch_id": batch_id,
                    "timeframe": timeframe
                },
                "X_features": {
                    "station": station_name,
                    "resumen_para_roberta": station_info['resumen_texto'],
                    "features_numericas_promedio": num_features
                },
                "Y_labels": y_labels
            }
            
            final_output.append(entry)

    save_json(final_output, OUTPUT_PATH)
    print(f"Successfully processed data. Output saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    process_batches()
