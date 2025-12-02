import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. 數學函數定義 (The Math) ---

def sigmoid(z):
    """
    Sigmoid 激活函數: 將任何數值壓縮到 0~1 之間 (機率)
    Formula: 1 / (1 + e^-z)
    """
    return 1 / (1 + np.exp(-z))

def compute_loss(y_true, y_pred):
    """
    二元交叉熵損失函數 (Binary Cross-Entropy Loss)
    衡量預測機率與真實標籤的差距
    """
    epsilon = 1e-15 # 避免 log(0) 發生錯誤
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def compute_gradients(X, y_true, y_pred):
    """
    計算梯度 (Gradients)
    這告訴我們參數 w 和 b 該往哪個方向調整才能降低 Loss
    Math: dL/dw = (1/m) * sum((y_pred - y_true) * x)
    """
    m = len(y_true)
    # 這裡的矩陣乘法 X.T.dot(...) 自動完成了 sigma 求和運算
    dw = (1 / m) * np.dot(X.T, (y_pred - y_true))
    db = (1 / m) * np.sum(y_pred - y_true)
    return dw, db

# --- 2. 數據準備 (Data Prep) ---

def load_and_preprocess_data(filepath):
    # 讀取數據
    df = pd.read_csv(filepath)
    
    # 選取特徵 (Features) 和 標籤 (Labels)
    X = df[['speed', 'error_rate']].values
    y = df['label_state'].values
    
    # 【關鍵步驟】數據標準化 (Normalization)
    # 因為 speed (約100) 和 error (約0.05) 數值範圍差太多
    # 如果不標準化，梯度下降會走得很慢甚至發散 (Zig-zag path)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std
    
    return X_scaled, y, mean, std

# --- 3. 訓練迴圈 (Training Loop) ---

def train_logistic_regression(X, y, learning_rate=0.1, epochs=1000):
    m, n = X.shape # m: 樣本數, n: 特徵數 (2)
    
    # 1. 初始化參數 (Weights & Bias)
    # 我們從零開始猜
    w = np.zeros(n) 
    b = 0
    
    loss_history = []
    
    print(f"開始訓練... (資料筆數: {m}, 特徵數: {n})")
    print("-" * 50)
    
    # 2. 梯度下降迴圈 (Gradient Descent Loop)
    for i in range(epochs):
        # A. 前向傳播 (Forward Pass): 預測
        z = np.dot(X, w) + b
        y_pred = sigmoid(z)
        
        # B. 計算損失 (Compute Loss)
        loss = compute_loss(y, y_pred)
        loss_history.append(loss)
        
        # C. 後向傳播 (Backward Pass): 算梯度
        dw, db = compute_gradients(X, y, y_pred)
        
        # D. 更新參數 (Update Parameters)
        # 這就是 "學習" 的瞬間: 往梯度的反方向走一步
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # 每 100 次迭代印出一次進度
        if i % 100 == 0:
            print(f"Epoch {i:4d} | Loss: {loss:.4f}")
            
    print("-" * 50)
    print(f"訓練完成! 最終 Loss: {loss:.4f}")
    print(f"學到的權重 (Weights): {w}")
    print(f"學到的偏差 (Bias): {b}")
    
    return w, b, loss_history

# --- 主程式 ---

if __name__ == "__main__":
    
    # --- 【關鍵修改】動態取得路徑 ---
    # 1. 取得目前這支程式的絕對路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 組合出 CSV 的完整路徑 (與程式碼在同一資料夾)
    # 確保您已經執行過第一步 generate_dataset.py
    data_filename = 'simulation_training_data.csv'
    data_path = os.path.join(current_dir, data_filename)
    
    print(f"正在讀取數據: {data_path}")
    
    # 3. 檢查檔案是否存在
    if not os.path.exists(data_path):
        print(f"\n[錯誤] 找不到檔案: {data_filename}")
        print("請確認：")
        print("1. 您是否已執行過第一步 (generate_dataset.py)？")
        print("2. 產生的 .csv 檔是否與本程式在同一個資料夾內？")
    else:
        # 4. 載入數據
        X, y, mean, std = load_and_preprocess_data(data_path)
        
        # 5. 開始訓練 (完全手刻版)
        w_final, b_final, history = train_logistic_regression(X, y, learning_rate=0.1, epochs=1000)
        
        # 6. 視覺化訓練過程
        plt.figure(figsize=(10, 5))
        plt.plot(history)
        plt.title('Training Process: Loss over Epochs')
        plt.xlabel('Iterations (Epochs)')
        plt.ylabel('Loss (Binary Cross Entropy)')
        plt.grid(True)
        
        # 圖片也存到相同資料夾
        plot_save_path = os.path.join(current_dir, 'logistic_regression_loss.png')
        plt.savefig(plot_save_path)
        print(f"\nLoss 曲線圖已儲存為: logistic_regression_loss.png")
        
        # 7. 解析學到的知識 (Interpretability)
        print("\n=== 模型解釋 (Model Interpretation) ===")
        print(f"特徵 1 (Speed) 的權重 w[0]: {w_final[0]:.4f}")
        print(f"特徵 2 (Error) 的權重 w[1]: {w_final[1]:.4f}")
        
        # 自動解釋權重意義
        if w_final[0] < 0:
            print(">> w[0] 為負值 (-)，代表『速度越快，挫折機率越低』。(符合直覺)")
        else:
            print(">> w[0] 為正值 (+)，代表『速度越快，挫折機率越高』。")
            
        if w_final[1] > 0:
            print(">> w[1] 為正值 (+)，代表『錯誤率越高，挫折機率越高』。(符合直覺)")
        else:
            print(">> w[1] 為負值 (-)，代表『錯誤率越高，挫折機率越低』。")