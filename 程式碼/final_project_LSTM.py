import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 用於畫漂亮的混淆矩陣
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# --- 1. 裝置設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"目前運算裝置: {device}")

# --- 2. 數據預處理 ---
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length), :2] 
        y = data[i + seq_length, 2]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# --- 3. 定義 LSTM 模型 ---
class FrustrationLSTM_Pro(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=1, num_layers=2):
        super(FrustrationLSTM_Pro, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)

# --- 4. 驗證與評估模組 (新增整合) ---
def run_verification(model, X_test, y_test, save_dir):
    """
    執行定量分析：計算準確率、混淆矩陣與分類報告
    """
    print("\n" + "="*40)
    print("正在執行 AI 模型健康檢查 (Verification)...")
    print("="*40)
    
    model.eval()
    X_test_device = X_test.to(device)
    
    with torch.no_grad():
        y_pred_prob = model(X_test_device).cpu().numpy()
        y_true = y_test.numpy()
    
    # 將機率轉為 0 或 1 (閾值 0.5)
    y_pred_class = (y_pred_prob > 0.5).astype(int)
    
    # A. 計算指標
    acc = accuracy_score(y_true, y_pred_class)
    print(f"總體準確率 (Accuracy): {acc*100:.2f}%")
    print("-" * 40)
    print("詳細指標 (Classification Report):")
    # target_names 對應 label 0 和 1
    print(classification_report(y_true, y_pred_class, target_names=['心流 (Flow)', '挫折 (Frustrated)']))
    
    # B. 繪製並儲存混淆矩陣
    cm = confusion_matrix(y_true, y_pred_class)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred: Flow', 'Pred: Frustrated'],
                yticklabels=['True: Flow', 'True: Frustrated'])
    plt.title('Confusion Matrix (Quantitative Verification)')
    plt.ylabel('Ground Truth')
    plt.xlabel('AI Prediction')
    
    matrix_path = os.path.join(save_dir, 'model_verification_matrix.png')
    plt.savefig(matrix_path)
    print(f"混淆矩陣圖已儲存為: {matrix_path}")
    print("="*40 + "\n")

# --- 5. 繪圖比較模組 ---
def plot_comparison(model, X_test, y_test, seq_length, save_dir):
    print("📊 正在繪製 LSTM vs LR 對比圖...")
    model.eval()
    X_test_device = X_test.to(device)
    
    with torch.no_grad():
        lstm_pred = model(X_test_device).cpu().numpy()
    
    # LR 預測 (Baseline)
    X_test_last_step = X_test[:, -1, :].numpy()
    w_lr = np.array([-3.12, 2.84])
    b_lr = 0.15
    z_lr = np.dot(X_test_last_step, w_lr) + b_lr
    lr_pred = 1 / (1 + np.exp(-z_lr))
    
    # 畫圖 (前 250 分鐘)
    limit = 250
    ground_truth = y_test[:limit].numpy()
    lstm_curve = lstm_pred[:limit]
    lr_curve = lr_pred[:limit]
    
    plt.figure(figsize=(14, 6))
    plt.fill_between(range(limit), ground_truth.flatten(), color='gray', alpha=0.15, label='Ground Truth (State)')
    plt.plot(lr_curve, label='Logistic Regression (No Memory)', linestyle='--', color='blue', alpha=0.4, linewidth=1)
    plt.plot(lstm_curve, label='LSTM Pro (Deep Memory)', linewidth=3, color='#d62728')
    
    plt.title(f'Final Project Result: Stability vs Sensitivity (Seq_Len={seq_length})')
    plt.ylabel('Frustration Probability')
    plt.xlabel('Time Step (Minutes)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    
    # 這裡就是您要求的固定檔名
    save_path = os.path.join(save_dir, 'LSTM_vs_LR_result.png')
    plt.savefig(save_path)
    print(f"比較圖表已儲存為: {save_path}")

# --- 6. 主訓練流程 ---
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'simulation_training_data.csv')
    
    if not os.path.exists(data_path):
        print("錯誤：找不到數據集！請先執行 generate_dataset.py")
        return

    # 讀取數據
    df = pd.read_csv(data_path)
    features = df[['speed', 'error_rate']].values
    labels = df['label_state'].values
    
    # 標準化
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    features_scaled = (features - mean) / std
    data_combined = np.column_stack((features_scaled, labels))
    
    # 參數設定
    SEQ_LENGTH = 30
    BATCH_SIZE = 128
    EPOCHS = 50
    LR = 0.005
    
    # 製作序列
    X_seq, y_seq = create_sequences(data_combined, SEQ_LENGTH)
    X_tensor = torch.FloatTensor(X_seq)
    y_tensor = torch.FloatTensor(y_seq).unsqueeze(1)
    
    # 切分數據 (80% 訓練, 20% 測試)
    train_size = int(len(X_tensor) * 0.8)
    X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
    y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]
    
    # DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # 初始化模型
    model = FrustrationLSTM_Pro(hidden_size=64, num_layers=2).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"啟動 LSTM Pro 訓練 (Seq: {SEQ_LENGTH}, Batch: {BATCH_SIZE})...")
    
    # 訓練迴圈
    for epoch in range(EPOCHS):
        model.train() 
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 5 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch:3d} | Avg Loss: {avg_loss:.4f}")
            
    print(f"訓練完成！")
    
    # --- 整合執行：先驗證數據，再畫對比圖 ---
    
    # 步驟 1: 執行定量驗證 (輸出 Accuracy 和 混淆矩陣)
    run_verification(model, X_test, y_test, current_dir)
    
    # 步驟 2: 執行定性繪圖 (輸出 LSTM_vs_LR_result.png)
    plot_comparison(model, X_test, y_test, SEQ_LENGTH, current_dir)

if __name__ == "__main__":
    main()