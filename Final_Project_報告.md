# 2025 ML Final Project: 個人化心智運算模型 

## 1. AI 的未來能力 (Future Capability)

### 核心願景：從「被動篩查」到「主動預防」

目前的 AI 僅能進行被動的問卷分析。我認為 20 年後，AI 將能實現**「個人化心智運算模型」**，為每個人建立**「數位心智雙生 (Digital Mind Twin)」**，深刻理解個人的**情緒慣性 (Emotional Inertia)** 與 **因果觸發點**。

### 應用場景

以社交焦慮患者為例：

- **預測 (Prediction)：** 綜合歷史與即時生理數據，預測 10 分鐘後焦慮指數將突破臨界值。
    
- **干預 (Intervention)：** 在崩潰前發送微型干預（如呼吸引導）或調整環境，實現「防患於未然」。
    
## 2. 所需成分 (Ingredients)

1. **資料 (Data)：多模態時序數據**（HRV、GSR、腦波、打字速度、語音特徵）。

2. **工具 (Tools)：** 因果推論 (Causal Inference) 找出焦慮源頭；圖神經網路 (GNN) 建構生活事件圖譜。

3. **學習架構 (Setup)：** 持續學習 (Lifelong Learning) 與 聯邦學習 (Federated Learning)。    

## 3. 機器學習類型 (ML Types)

本階段採混合策略，以 **監督式學習 (Supervised Learning)** 為核心驗證手段。

- **目的：** 驗證「客觀行為 ($X$)」與「主觀狀態 ($Y$)」的映射關係。
    
- **未來擴展：** 引入 **非監督學習** 解決標籤稀缺問題；引入 **強化學習** 優化最佳干預時機。

## 4. 第一步 : 可實作模型問題 (Solvable Model Problem)

**目標：** 開發 **「學習者挫折偵測系統」**，驗證能否透過行為數據推論內在狀態。

### 4.1 問題與理論對應

- **輸入 (**$X$**)：** 打字速度 (Speed)、錯誤率 (Error Rate) 的時間序列。
    
- **輸出 (**$Y$**)：** 狀態判斷 (0: 心流, 1: 挫折)。
	
- **理論對應：**
    
    1. **情緒慣性理論：** 精神狀態具有「自我維持」特性。我們使用 HMM 生成具備慣性的模擬數據。
        
    2. **上下文依賴：** 單一行為的意義取決於上下文。我們引入 LSTM 的記憶單元來模擬此機制。

### 4.2 模型與方法演進 (Model Design)

我們採用「從規則到深度學習」的漸進式路線：

| 特性     | Baseline (邏輯回歸)              | Main Model (LSTM Pro)               |
| ------ | ---------------------------- | ----------------------------------- |
| **架構** | 單一神經元 (無記憶)                  | 雙層 LSTM (具備長短期記憶)                   |
| **假設** | 時間點獨立 ($y_t = \sigma(Wx_t)$) | 時間相依 ($y_t = LSTM(x_{t-30}...x_t)$) |
| **目的** | 驗證特徵線性相關性                    | **捕捉情緒慣性，提升抗噪能力**                   |
<table> <tr> <td align="center" width="50%"><strong>Baseline 架構 (單神經元)</strong></td> <td align="center" width="50%"><strong>LSTM 架構 (時序記憶)</strong></td> </tr> <tr> <td><img src="Baseline Model架構(Logistic Regression).png" width="100%"></td> <td><img src="Main Model架構(LSTM).png" width="100%"></td> </tr> </table>


### 4.3 實作與結果 (Implementation & Results)

我們使用 PyTorch 實作並生成 50,000 筆模擬數據，測試集驗證結果如下：

#### A. Baseline 訓練結果 (Logistic Regression)

邏輯回歸成功收斂，權重 ($w_{speed} \approx -3.12, w_{error} \approx +2.84$) 符合直覺：速度越快越不像挫折，錯誤越多越像挫折。(左圖：Loss 曲線 / 右圖：訓練權重輸出)
<table> <tr> <td><img src="logistic_regression_loss.png" width="100%"></td> <td><img src="code_result_LR.png" width="100%"></td> </tr> </table>
#### B. LSTM 定量評估

在 9,994 筆測試資料中，LSTM 達到 **90.26% Accuracy**。混淆矩陣顯示誤報率與漏報率皆低於 5%。
<table> <tr> <td align="center"><strong>終端機執行結果 (準確率鐵證)</strong></td> <td align="center"><strong>混淆矩陣 (無偏差驗證)</strong></td> </tr> <tr> <td><img src="code_result_LSTM.png" width="100%"></td> <td><img src="model_verification_matrix.png" width="100%"></td> </tr> </table>

#### C. 關鍵對比：抗噪能力 (Noise Robustness)

下圖為兩模型在同一時間序列上的預測對比：
<img src="LSTM_vs_LR_result.png" width="80%">


- **藍線 (Baseline)：** 在灰色挫折區間 ($t=60$) 出現劇烈閃爍 (Flickering)，因短期行為雜訊而誤判。
    
- **紅線 (LSTM)：** 展現極佳穩定性。憑藉記憶機制，它能過濾短期雜訊，穩準地判斷使用者仍處於挫折狀態。

### 4.4 討論 (Discussion)

本專案證實了 **「精神狀態偵測必須納入時間維度」**。 單純的行為特徵雖有預測力，但缺乏「上下文」易導致誤報。LSTM 透過記憶單元有效模擬了情緒慣性，成功解決了 Baseline 模型「見樹不見林」的缺陷，為未來開發高可靠度的個人化心智模型奠定了基礎。

### 附錄：程式碼結構說明

* **`final_project_data.py`**: HMM 數據生成器。
* **`final_project_LogisticRegression.py`**: Baseline (手刻 LR) 訓練與推導。
* **`final_project_LSTM.py`**: Main Model (PyTorch LSTM) 訓練、驗證與繪圖整合。





