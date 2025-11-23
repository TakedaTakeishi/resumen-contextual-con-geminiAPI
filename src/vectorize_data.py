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
    y_types = [item['Y_labels']['tipo_falla'] for item in data]
    y_modes = [item['Y_labels']['modo_falla_codigo'] for item in data]
    y_impacts = [item['Y_labels']['impacto_operativo'] for item in data]
    
    # Encoders
    le_station = LabelEncoder()
    le_type = LabelEncoder()
    le_mode = LabelEncoder()
    le_impact = LabelEncoder()
    
    # Fit encoders (handle None for Y labels)
    station_encoded = le_station.fit_transform(stations)
    
    # For Y labels, we need to handle None. We'll convert None to 'None' string for encoding
    y_types_str = [str(x) for x in y_types]
    y_modes_str = [str(x) for x in y_modes]
    y_impacts_str = [str(x) for x in y_impacts]
    
    type_encoded = le_type.fit_transform(y_types_str)
    mode_encoded = le_mode.fit_transform(y_modes_str)
    impact_encoded = le_impact.fit_transform(y_impacts_str)
    
    # Save mappings
    mappings = {
        "stations": dict(zip(le_station.classes_, le_station.transform(le_station.classes_))),
        "tipo_falla": dict(zip(le_type.classes_, le_type.transform(le_type.classes_))),
        "modo_falla_codigo": dict(zip(le_mode.classes_, le_mode.transform(le_mode.classes_))),
        "impacto_operativo": dict(zip(le_impact.classes_, le_impact.transform(le_impact.classes_)))
    }
    save_json(mappings, MAPPINGS_PATH)
    
    # 4. Construct Final JSON
    print("Constructing final dataset...")
    final_dataset = []
    
    for i, item in enumerate(data):
        # Get scaled numerical features
        num_feats = df_num_scaled.iloc[i].to_dict()
        
        final_entry = {
            "batch_id": item['batch_meta']['batch_id'],
            "timeframe": item['batch_meta']['timeframe'],
            "station_id": int(station_encoded[i]),
            "station_name": stations[i],
            
            # X Features
            "X_embedding": embeddings[i].tolist(), # The vector
            "X_numerical_normalized": num_feats,
            
            # Y Labels (Encoded)
            "Y_falla_detectada": int(item['Y_labels']['falla_detectada']),
            "Y_tipo_falla_encoded": int(type_encoded[i]),
            "Y_modo_falla_encoded": int(mode_encoded[i]),
            "Y_impacto_encoded": int(impact_encoded[i]),
            "Y_severidad": item['Y_labels']['severidad'],
            "Y_retraso_minutos": item['Y_labels']['retraso_minutos']
        }
        final_dataset.append(final_entry)
        
    save_json(final_dataset, OUTPUT_PATH)
    print(f"Done! Final dataset saved to {OUTPUT_PATH}")
    print(f"Mappings saved to {MAPPINGS_PATH}")

if __name__ == "__main__":
    process_vectorization()
