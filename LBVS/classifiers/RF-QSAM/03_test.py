import os
import pandas as pd
import numpy as np
import joblib  # 替换keras模型加载

def get_trained_models():
    model_info = []
    for i in range(1, 11):
        subset = f"subset_{i:02d}"
        model_path = os.path.join("save_models", subset, "final_model.pkl")
        if os.path.exists(model_path):
            model_info.append((model_path, subset))
    return model_info

def predict_and_average(dataset_name, model_info_list):
    """对指定数据集使用所有模型预测并计算平均值"""
    processed_path = os.path.join("data", dataset_name, f"{dataset_name}_processed.csv")
    if not os.path.exists(processed_path):
        print(f"错误：{processed_path} 不存在")
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
        print(f"加载 {dataset_name} 失败: {e}")
        return

    all_predictions = []
    for model_path, subset_name in model_info_list:
        try:
            model = joblib.load(model_path) 
            y_pred_proba = model.predict_proba(features)[:, 1]
            # 新增：将当前模型的预测结果添加到列表
            all_predictions.append(y_pred_proba)  
        except Exception as e:
            print(f"模型 {subset_name} 处理失败: {e}")
            continue

    if not all_predictions:
        print(f"{dataset_name} 无有效预测结果")
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
    print(f"保存 {dataset_name} 结果至 {output_path}")

if __name__ == "__main__":
    model_info_list = get_trained_models()
    
    datasets_to_predict = ["QSIDB"]
    for dataset in datasets_to_predict:
        predict_and_average(dataset, model_info_list)
    
    print("所有处理完成")