import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import json
import joblib

def binary_cross_entropy(y_true, y_pred_proba, epsilon=1e-10):
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon) 
    return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))

def build_model():
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=30,    
        min_samples_split=2,  
        min_samples_leaf=1, 
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    return model

def train_model(train_data_path, model_save_dir, log_path, threshold=0.5): 
    
    os.makedirs(model_save_dir, exist_ok=True)
    
    df = pd.read_csv(train_data_path, low_memory=False)
    fp_columns = [col for col in df.columns if col.startswith('fp_')]
    fp_df = df[fp_columns].copy()   

    for col in fp_df.columns:
        fp_df[col] = pd.to_numeric(fp_df[col], errors='coerce')
    fp_df.fillna(0, inplace=True)
    
    X_train = fp_df.values.astype(np.float32)
    y_train = df['label'].values.astype(np.float32)
    
    model = build_model()

    # Train model
    model.fit(X_train, y_train)
    
    # Calculate training loss
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    train_loss = binary_cross_entropy(y_train, y_train_pred_proba)
    
    # Save model parameters and metrics
    model_params = {
        'model_type': 'RandomForest',
        'n_estimators': 100,
        'max_depth': 30,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'threshold': threshold,
        'train_samples': len(X_train),
        'train_loss': train_loss
    }
    with open(os.path.join(model_save_dir, 'model_params.json'), 'w') as f:
        json.dump(model_params, f, indent=4)

    print(f"Train Loss: {train_loss:.4f}")
    
    # Save model
    joblib.dump(model, os.path.join(model_save_dir, "final_model.pkl"))
    print(f"Training completed! Model saved as final_model.pkl, train_loss: {train_loss:.4f}")

    log_df = pd.DataFrame([{'train_loss': train_loss}])
    log_df.to_csv(log_path, index=False)

if __name__ == "__main__":
    subsets = ["subset_01", "subset_02", "subset_03", "subset_04", "subset_05", 
               "subset_06", "subset_07", "subset_08", "subset_09", "subset_10"]
    
    for subset in subsets:
        print(f"\n===== Starting training for {subset} =====")
        train_data_path = os.path.join("data", "base_train", subset, f"{subset}_processed.csv")
        model_save_dir = os.path.join("save_models", subset)
        log_path = os.path.join(model_save_dir, "train_log.csv")
        
        train_model(
            train_data_path=train_data_path,
            model_save_dir=model_save_dir,
            log_path=log_path,
            threshold=0.5
        )
    print("\nAll datasets training completed!")