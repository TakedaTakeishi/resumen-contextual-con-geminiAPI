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
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'DATOS_CRUDOS_L1_COMPLETO_REORGANIZADOS.json')
REPORTS_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'INFORMES_SIEMONS_DATOS_CRUDOS_L1_COMPLETOS_REORGANIZADOS_REAL.json')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'features.json')

# Configuration
# BATCH_SIZE is no longer fixed, determined by reports
INCLUDE_AUX_INFO = False

# Set API Key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")

genai.configure(api_key=api_key)

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def parse_batch_range(range_str):
    """
    Parses a string like "1-200" or "Registros 1-200" into start and end indices (0-based).
    """
    # Extract numbers
    nums = re.findall(r'\d+', range_str)
    if len(nums) >= 2:
        start = int(nums[0]) - 1 # Convert to 0-based index
        end = int(nums[1])
        return start, end
    return None, None

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
    
    retries = 3
    for attempt in range(retries):
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
                
                print(f"Warning: No JSON found in response (Attempt {attempt+1}/{retries}). Start: {text[:50]}...")
                
        except Exception as e:
            print(f"Error calling Gemini (Attempt {attempt+1}/{retries}): {e}")
            time.sleep(2) # Wait before retry
            
    return []

def process_batches():
    print(f"Loading raw data from: {RAW_DATA_PATH}")
    raw_data = load_json(RAW_DATA_PATH)
    print(f"Loading reports from: {REPORTS_PATH}")
    reports_data = load_json(REPORTS_PATH)
    
    reports_list = reports_data.get('reports', [])
    if not reports_list:
        print("No reports found in the reports file.")
        return

    final_output = []
    
    total_records = len(raw_data)
    print(f"Total raw records: {total_records}")
    print(f"Processing {len(reports_list)} batches defined in reports...")

    for i, report in enumerate(reports_list):
        batch_id = report.get('batch_id')
        batch_range_str = report.get('batch_records')
        
        if not batch_range_str:
            print(f"Skipping batch {batch_id}: No batch_records defined.")
            continue
            
        start_idx, end_idx = parse_batch_range(batch_range_str)
        
        if start_idx is None or end_idx is None:
            print(f"Skipping batch {batch_id}: Could not parse range '{batch_range_str}'.")
            continue
            
        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(total_records, end_idx)
        
        if start_idx >= end_idx:
            print(f"Skipping batch {batch_id}: Invalid range {start_idx}-{end_idx}.")
            continue

        batch_data = raw_data[start_idx:end_idx]
        print(f"Processing Batch {batch_id} (Records {start_idx+1}-{end_idx}, Count: {len(batch_data)})...")
        
        # 1. Gemini Extraction
        extracted_stations = get_gemini_extraction(batch_data)
        
        # Sleep briefly to avoid hitting rate limits too hard
        time.sleep(2)
        
        timeframe = report.get('batch_timeframe', 'Unknown')
        
        # If the report mentions a specific station where a failure occurred, 
        # we MUST ensure it's in our extracted list (even if our simple mock missed it, 
        # though the mock should catch it if it's in the text).
        # For the purpose of the dataset generation, we iterate over what we "found".
        
        # Also, if the report says "falla_detectada": true, we want to make sure we align 
        # the Y_labels correctly for the station where it happened.
        
        if not extracted_stations:
             print(f"  Warning: No stations extracted for batch {batch_id}.")

        for station_info in extracted_stations:
            station_name = station_info['station_id']
            
            # 2. Numerical Features Extraction
            num_features = extract_numerical_features(batch_data, station_name)
            
            # 3. Construct Y_labels
            y_labels = {
                "falla_detectada": False,
                "target_falla": None
            }
            
            # Auxiliary info for RPN calculation and metadata, kept outside Y_labels as requested
            aux_info = {
                "modo_falla_codigo": None,
                "impacto_operativo": "Normal",
                "retraso_minutos": 0,
                "descripcion_tecnica": None,
                "severidad": 0
            }
            
            if report.get('falla_detectada'):
                # Check if this station matches the one in the report
                report_location = report.get('location_segment')
                # Robust matching: check if one string contains the other (case insensitive)
                if report_location and (report_location.lower() in station_name.lower() or station_name.lower() in report_location.lower()):
                    y_labels["falla_detectada"] = True
                    y_labels["target_falla"] = report.get('root_cause_category')
                    
                    aux_info["modo_falla_codigo"] = report.get('failure_mode_code')
                    aux_info["impacto_operativo"] = report.get('operational_impact')
                    aux_info["retraso_minutos"] = report.get('delay_minutes')
                    aux_info["descripcion_tecnica"] = report.get('incident_description')
                    
                    # Map severity
                    impact = report.get('operational_impact', '')
                    if 'Suspensión' in impact:
                        aux_info["severidad"] = 2
                    elif 'Retraso' in impact:
                        aux_info["severidad"] = 1
                    else:
                        aux_info["severidad"] = 0

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

            if INCLUDE_AUX_INFO:
                entry["auxiliary_info"] = aux_info
            
            final_output.append(entry)

    save_json(final_output, OUTPUT_PATH)
    print(f"Successfully processed data. Output saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    process_batches()
