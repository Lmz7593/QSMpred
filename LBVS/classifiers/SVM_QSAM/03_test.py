import os
import pandas as pd
import numpy as np
import joblib

def get_trained_models():
    """获取所有训练好的模型"""
    model_info = []
    for i in range(1, 11):
        subset = f"subset_{i:02d}"
        model_path = os.path.join("save_models", subset, "final_model.pkl")
        scaler_path = os.path.join("save_models", subset, "scaler.pkl")
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model_info.append((model_path, scaler_path, subset))
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
    
    for model_path, scaler_path, subset_name in model_info_list:
        try:
            # 加载模型和标准化器
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            # 特征标准化
            features_scaled = scaler.transform(features)
            
            # 预测概率
            y_pred_proba = model.predict_proba(features_scaled)[:, 1]
            all_predictions.append(y_pred_proba)
            
            print(f"模型 {subset_name} 预测完成")
            
        except Exception as e:
            print(f"模型 {subset_name} 处理失败: {e}")
            continue

    if not all_predictions:
        print(f"{dataset_name} 无有效预测结果")
        return

    # 计算平均预测概率
    avg_pred_proba = np.mean(all_predictions, axis=0)
    
    # 保存结果
    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)
    output_path = os.path.join(result_dir, f"{dataset_name}.csv")
    
    result_df = pd.DataFrame({
        'ID': df['ID'],
        'y_pred_proba_avg': avg_pred_proba
    })
    result_df.to_csv(output_path, index=False)
    
    print(f"保存 {dataset_name} 结果至 {output_path}")
    print(f"平均预测概率统计 - 均值: {np.mean(avg_pred_proba):.4f}, 标准差: {np.std(avg_pred_proba):.4f}")

if __name__ == "__main__":
    model_info_list = get_trained_models()
    
    if not model_info_list:
        print("未找到训练好的模型，请先运行训练脚本")
    else:
        print(f"找到 {len(model_info_list)} 个SVM模型")
        
        datasets_to_predict = ["QSIDB"]
        for dataset in datasets_to_predict:
            print(f"\n----- 处理 {dataset} -----")
            predict_and_average(dataset, model_info_list)
    
    print("\n所有处理完成")