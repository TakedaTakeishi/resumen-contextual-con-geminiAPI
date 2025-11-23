import json
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'features.json')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_final.json')
MAPPINGS_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'mappings.json')

# Failure Mode Metadata (RPN)
FAILURE_MODE_METADATA = {
    "ENRG-001": {"description": "Pérdida total de alimentación", "rpn": 10},
    "ENRG-002": {"description": "Caída de tensión", "rpn": 8},
    "ENRG-003": {"description": "Sobretensión", "rpn": 9},
    "ENRG-004": {"description": "Cortocircuito en catenaria", "rpn": 8},
    "ENRG-005": {"description": "Pérdida de contacto pantógrafo", "rpn": 7},
    "TRAC-001": {"description": "Motor bloqueado", "rpn": 9},
    "TRAC-002": {"description": "Pérdida de par motor", "rpn": 8},
    "TRAC-003": {"description": "Falla de inversor", "rpn": 8},
    "TRAC-004": {"description": "Degradación de frenado", "rpn": 9},
    "MECH-001": {"description": "Puerta no cierra", "rpn": 7},
    "MECH-002": {"description": "Puerta no abre", "rpn": 9},
    "MECH-003": {"description": "Fuga de aire comprimido", "rpn": 8},
    "MECH-004": {"description": "Rodamiento sobrecalentado", "rpn": 9},
    "MECH-005": {"description": "Suspensión colapsada", "rpn": 8},
    "SIGN-001": {"description": "Circuito de vía en falso ocupado", "rpn": 7},
    "SIGN-002": {"description": "Pérdida de comunicación ATP", "rpn": 9},
    "SIGN-003": {"description": "Enclavamiento bloqueado", "rpn": 8}
}

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    # Convert numpy types to python types for JSON serialization
    def default(obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
        
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=default)

def process_vectorization():
    print("Loading data...")
    data = load_json(INPUT_PATH)
    
    # 1. Vectorization (XLM-RoBERTa)
    print("Loading XLM-RoBERTa model...")
    # Using a lightweight multilingual model based on XLM-R
    model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
    
    texts = [item['X_features']['resumen_para_roberta'] for item in data]
    print(f"Vectorizing {len(texts)} summaries...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # 2. Numerical Normalization
    print("Normalizing numerical features...")
    # Extract numerical features into a DataFrame
    num_features_list = []
    for item in data:
        feats = item['X_features']['features_numericas_promedio']
        # Handle None values (impute with 0 or mean - here 0 for simplicity/safety)
        clean_feats = {
            'temp': feats.get('temp') or 0,
            'humidity': feats.get('humidity') or 0,
            'precip_mm': feats.get('precip_mm') or 0,
            'traffic_jam_level': feats.get('traffic_jam_level') or 0
        }
        num_features_list.append(clean_feats)
    
    df_num = pd.DataFrame(num_features_list)
    scaler = StandardScaler()
    df_num_scaled = pd.DataFrame(scaler.fit_transform(df_num), columns=df_num.columns)
    
    # 3. Categorical Encoding
    print("Encoding categorical variables...")
    
    # Collect all categorical values
    stations = [item['X_features']['station'] for item in data]
    
    # Y labels
    y_types = [item['Y_labels']['target_falla'] for item in data]
    y_modes = [item.get('auxiliary_info', {}).get('modo_falla_codigo') for item in data]
    y_impacts = [item.get('auxiliary_info', {}).get('impacto_operativo') for item in data]
    
    # Encoders
    le_station = LabelEncoder()
    le_type = LabelEncoder()
    le_mode = LabelEncoder()
    le_impact = LabelEncoder()
    
    # Fit encoders (handle None for Y labels)
    station_encoded = le_station.fit_transform(stations)
    
    # For Y labels, we need to handle None. We'll convert None to 'None' string for encoding
    y_types_str = [str(x) for x in y_types]
    
    # Ensure all known failure modes are in the encoder classes, plus 'None'
    known_modes = list(FAILURE_MODE_METADATA.keys()) + ['None']
    # Combine with actual data to ensure we don't miss anything unexpected
    all_modes = list(set(known_modes + [str(x) for x in y_modes]))
    le_mode.fit(all_modes)
    
    y_modes_str = [str(x) for x in y_modes]
    y_impacts_str = [str(x) for x in y_impacts]
    
    type_encoded = le_type.fit_transform(y_types_str)
    mode_encoded = le_mode.transform(y_modes_str)
    impact_encoded = le_impact.fit_transform(y_impacts_str)
    
    # Save mappings
    # Convert numpy types to native python types for JSON serialization
    mappings = {
        "stations": {str(k): int(v) for k, v in zip(le_station.classes_, le_station.transform(le_station.classes_))},
        "target_falla": {str(k): int(v) for k, v in zip(le_type.classes_, le_type.transform(le_type.classes_))},
        "rpn_metadata": FAILURE_MODE_METADATA
    }
    save_json(mappings, MAPPINGS_PATH)
    
    # 4. Construct Final JSON
    print("Constructing final dataset...")
    final_dataset = []
    
    for i, item in enumerate(data):
        # Get scaled numerical features
        num_feats = df_num_scaled.iloc[i].to_dict()
        
        # Get RPN
        mode_code = item.get('auxiliary_info', {}).get('modo_falla_codigo')
        rpn_value = 0
        if mode_code in FAILURE_MODE_METADATA:
            rpn_value = FAILURE_MODE_METADATA[mode_code]['rpn']
        
        final_entry = {
            "batch_id": item['batch_meta']['batch_id'],
            "timeframe": item['batch_meta']['timeframe'],
            "station_id": int(station_encoded[i]),
            
            # X Features
            "X_embedding": embeddings[i].tolist(), # The vector
            "X_numerical_normalized": num_feats,
            
            # Y Labels (Encoded)
            "target_falla": int(type_encoded[i]),
            "Y_RPN": rpn_value
        }
        final_dataset.append(final_entry)
        
    save_json(final_dataset, OUTPUT_PATH)
    print(f"Done! Final dataset saved to {OUTPUT_PATH}")
    print(f"Mappings saved to {MAPPINGS_PATH}")

if __name__ == "__main__":
    process_vectorization()
