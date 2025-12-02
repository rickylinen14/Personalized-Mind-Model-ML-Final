# Learner Frustration Detection System (LSTM vs. Logistic Regression)

### 2025 Machine Learning Final Project

## 📖 專案簡介 (Project Overview)

本專案旨在驗證 **「個人化心智運算模型 (Personalized Computational Model of the Mind)」** 的核心概念：我們能否僅透過客觀的行為數據（如打字速度、錯誤率），推論出使用者內在的主觀狀態（如心流或挫折）？

為了回答這個問題，我們設計了一個模擬實驗，並比較了 **無記憶模型 (Logistic Regression)** 與 **時序記憶模型 (LSTM)** 的表現。結果證實，引入深度學習的記憶機制對於捕捉精神狀態中的「情緒慣性 (Emotional Inertia)」至關重要。

## 📂 檔案結構 (File Structure)

本 Repository 包含以下核心檔案，依功能分類：

### 核心程式碼 (Source Code)

- **`final_project_data.py`**
    
    - **功能：** 基於隱馬可夫模型 (HMM) 邏輯的數據生成器。
        
    - **輸出：** 執行後會自動產生 50,000 筆包含隨機特徵的訓練數據 (`simulation_training_data.csv`)。
        
- **`final_project_LogisticRegression.py`**
    
    - **功能：** **從零手刻 (From Scratch)** 的邏輯回歸算法。包含 Sigmoid、Cross-Entropy Loss 與 Gradient Descent 的數學實作。
        
    - **目的：** 作為 Baseline，驗證特徵與狀態之間的線性相關性。
        
- **`final_project_LSTM.py`**
    
    - **功能：** 使用 **PyTorch** 構建的雙層 LSTM 模型。
        
    - **特色：** 包含 Mini-batch 訓練、自動硬體加速 (CPU/GPU)、定量驗證 (混淆矩陣) 與定性繪圖。
        
    - **目的：** 捕捉時間序列中的長短期依賴關係，提升抗噪能力。
        

### 報告與文件 (Documentation)

- **`Final_Project_報告.pdf`** (或是 .md)
    
    - **內容：** 完整的期末專案報告，包含問題描述、理論對應、模型設計、結果分析與討論。
        
- **`requirements.txt`**
    
    - **內容：** 專案所需的 Python 套件列表。
-  **`images`**
    
    - 存放報告與 README 使用的實驗結果截圖與架構圖。
        

## 📊 實驗結果亮點 (Results)

- **準確率 (Accuracy)：** LSTM 模型在測試集上達到 **90.26%**。
    
- **抗噪能力 (Robustness)：** 如下圖所示，LSTM (紅線) 相比於 Baseline (藍線)，能有效過濾短期行為雜訊，展現極佳的預測穩定性。![Comparison](images/LSTM_vs_LR_result.png)
    

## 📑 專案報告 (Project Report)

詳細的理論推導、模型架構圖以及深度討論，請參閱完整的專案報告：Final_Project_報告.md/pdf

## 🚀 如何執行 (How to Run)

請依序執行以下指令以重現實驗結果：

**1. 安裝依賴套件**

```
pip install -r requirements.txt
```

**2. 生成模擬數據**

```
python final_project_data.py
```

_(此步驟會生成 simulation_training_data.csv)_

**3. 訓練 Baseline 模型 (可選)**

```
python final_project_LogisticRegression.py
```

_(將輸出 Loss 曲線與權重訓練結果)_

**4. 訓練 LSTM 模型並產出最終報告圖表**

```
python final_project_LSTM.py
```

_(程式將自動進行訓練、驗證，並輸出 `LSTM_vs_LR_result.png` 與 `model_verification_matrix.png`)_

## 📦 其他檔案說明 (Others)

- **`simulation_training_data.csv`**
    
    - 這是由 `final_project_data.py` 生成的模擬數據集。若您不想自行生成，可直接使用此檔案進行訓練。
        
    - **格式：** CSV
        
    - **內容：** 包含 `time_step`, `speed`, `error_rate`, `label_state`, `session_id` 等欄位。
        
- **圖片檔案 (*.png)**
    
    - 包含 `code_result.png` (執行結果截圖)、`model_verification_matrix.png` (混淆矩陣) ...... 等，皆為報告中使用的素材或程式輸出的結果。
        

Created by [林品睿/314611018] for 2025 Machine Learning Course.