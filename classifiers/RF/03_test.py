import os
import pandas as pd
import numpy as np
import joblib

def get_trained_models():
    model_info = []
    for i in range(1, 11):
        subset = f"subset_{i:02d}"
        model_path = os.path.join("save_models", subset, "final_model.pkl")
        if os.path.exists(model_path):
            model_info.append((model_path, subset))
    return model_info

def predict_and_average(dataset_name, model_info_list):
    """Use all models to predict on the specified dataset and calculate the average"""
    processed_path = os.path.join("data", dataset_name, f"{dataset_name}_processed.csv")
    if not os.path.exists(processed_path):
        print(f"Error: {processed_path} does not exist")
        return

    try:
        df = pd.read_csv(processed_path, low_memory=False)
        fp_columns = [col for col in df.columns if col.startswith('fp_')]
        fp_df = df[fp_columns].copy()
        for col in fp_df.columns:
            fp_df[col] = pd.to_numeric(fp_df[col], errors='coerce')
        fp_df.fillna(0, inplace=True)
        features = fp_df.values.astype(np.float32)
    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")
        return

    all_predictions = []
    for model_path, subset_name in model_info_list:
        try:
            model = joblib.load(model_path) 
            y_pred_proba = model.predict_proba(features)[:, 1]
            all_predictions.append(y_pred_proba)  
        except Exception as e:
            print(f"Model {subset_name} processing failed: {e}")
            continue

    if not all_predictions:
        print(f"{dataset_name} has no valid prediction results")
        return

    avg_pred_proba = np.mean(all_predictions, axis=0)
    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)
    output_path = os.path.join(result_dir, f"{dataset_name}.csv")
    result_df = pd.DataFrame({
        'ID': df['ID'],
        'y_pred_proba_avg': avg_pred_proba
    })
    result_df.to_csv(output_path, index=False)
    print(f"Results for {dataset_name} saved to {output_path}")

if __name__ == "__main__":
    model_info_list = get_trained_models()
    
    datasets_to_predict = ["QSIDB", "DrugBank", "MiMeDB"]
    for dataset in datasets_to_predict:
        predict_and_average(dataset, model_info_list)
    
    print("All processing completed")