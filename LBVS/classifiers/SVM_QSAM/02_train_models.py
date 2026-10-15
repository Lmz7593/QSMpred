import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import json
import joblib

def binary_cross_entropy(y_true, y_pred_proba, epsilon=1e-10):
    """相同的损失函数"""
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon) 
    return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))

def build_svm_model():
    """构建SVM模型"""
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42,
        cache_size=1000
    )
    return model

def optimize_svm_parameters(X_train, y_train):
    """SVM参数优化（可选）"""
    print("正在进行SVM参数优化...")
    
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1, 0.01],
    }
    
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    grid_search = GridSearchCV(
        svm, param_grid, cv=3, scoring='neg_log_loss', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳分数: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_svm_model(train_data_path, model_save_dir, log_path, threshold=0.5, optimize_params=False):
    
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 读取数据
    df = pd.read_csv(train_data_path, low_memory=False)
    fp_columns = [col for col in df.columns if col.startswith('fp_')]
    fp_df = df[fp_columns].copy()   
    
    # 数据预处理
    for col in fp_df.columns:
        fp_df[col] = pd.to_numeric(fp_df[col], errors='coerce')
    fp_df.fillna(0, inplace=True)
    
    X_train = fp_df.values.astype(np.float32)
    y_train = df['label'].values.astype(np.float32)
    
    # 特征标准化 - SVM对特征尺度敏感
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 构建和训练模型
    if optimize_params:
        model = optimize_svm_parameters(X_train_scaled, y_train)
    else:
        model = build_svm_model()
        model.fit(X_train_scaled, y_train)
    
    # 计算训练损失
    y_train_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
    train_loss = binary_cross_entropy(y_train, y_train_pred_proba)
    
    # 保存模型参数
    model_params = {
        'model_type': 'SVM',
        'kernel': model.kernel,
        'C': model.C,
        'gamma': str(model.gamma),
        'probability': model.probability,
        'threshold': threshold,
        'train_samples': len(X_train),
        'train_loss': train_loss,
        'feature_dimension': X_train.shape[1]
    }
    
    with open(os.path.join(model_save_dir, 'model_params.json'), 'w') as f:
        json.dump(model_params, f, indent=4)
    
    # 保存模型和标准化器
    joblib.dump(model, os.path.join(model_save_dir, "final_model.pkl"))
    joblib.dump(scaler, os.path.join(model_save_dir, "scaler.pkl"))
    
    print(f"SVM Train Loss: {train_loss:.4f}")
    print(f"训练完成！SVM模型已保存，train_loss: {train_loss:.4f}")
    
    # 保存日志
    log_df = pd.DataFrame([{'train_loss': train_loss}])
    log_df.to_csv(log_path, index=False)

if __name__ == "__main__":
    subsets = ["subset_01", "subset_02", "subset_03", "subset_04", "subset_05", 
               "subset_06", "subset_07", "subset_08", "subset_09", "subset_10"]
    
    for subset in subsets:
        print(f"\n===== 开始训练SVM {subset} =====")
        train_data_path = os.path.join("data", "base_train", subset, f"{subset}_processed.csv")
        model_save_dir = os.path.join("save_models", subset)
        log_path = os.path.join(model_save_dir, "train_log.csv")
        
        train_svm_model(
            train_data_path=train_data_path,
            model_save_dir=model_save_dir,
            log_path=log_path,
            threshold=0.5,
            optimize_params=False
        )
    print("\n所有数据集SVM训练完成！")