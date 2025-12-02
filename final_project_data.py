import numpy as np
import pandas as pd
import os

# --- 引入您原本的模擬器邏輯 ---
# 為了產生大量數據，我們稍微封裝一下之前的 StrictStudent
class DataGenerator:
    def __init__(self):
        self.state = 0 # 0:Flow, 1:Frustrated
        
    def generate_session(self, steps=100):
        # 這裡我們隨機化參數，讓數據集更多元 (Data Augmentation 的概念)
        # 讓每位"虛擬學生"的抗壓性都不太一樣
        fall_prob = np.random.uniform(0.05, 0.15)
        recover_prob = np.random.uniform(0.05, 0.15) # 沒 AI 的自然恢復率
        
        data = []
        for t in range(steps):
            # 狀態轉移
            if self.state == 0:
                if np.random.rand() < fall_prob: self.state = 1
            else:
                if np.random.rand() < recover_prob: self.state = 0
            
            # 觀測數據
            if self.state == 0:
                speed = np.random.normal(100, 10)
                error = np.random.beta(1, 20)
            else:
                speed = np.random.normal(40, 15)
                error = np.random.beta(5, 5)
                
            data.append({
                "time_step": t,
                "speed": speed,
                "error_rate": error,
                "label_state": self.state # 這是 Ground Truth，訓練時的答案
            })
        return data

# --- 生成訓練資料集 ---
def create_dataset(num_sessions=100, steps_per_session=100):
    all_data = []
    print(f"正在生成 {num_sessions} 位學生的模擬數據...")
    
    for i in range(num_sessions):
        student = DataGenerator()
        session_data = student.generate_session(steps_per_session)
        # 加上 session_id 以便區分不同學生
        for row in session_data:
            row['session_id'] = i
        all_data.extend(session_data)
        
    df = pd.DataFrame(all_data)
    return df

if __name__ == "__main__":
    # 生成 500 位學生，每人跑 100 分鐘的數據 -> 總共 50,000 筆資料
    df_dataset = create_dataset(num_sessions=500, steps_per_session=100)

    current_file_path = os.path.abspath(__file__)

    current_dir_path = os.path.dirname(current_file_path)
    
    # 存成 CSV 檔，這就是您期末 ML 模型的 "飼料"
    save_filename = 'simulation_training_data.csv'
    save_path = os.path.join(current_dir_path, save_filename)
    df_dataset.to_csv(save_path, index=False)
    
    print(f"數據集已生成！包含 {len(df_dataset)} 筆資料。")
    print(f"前 5 筆資料預覽：")
    print(df_dataset.head())
    print(f"\n檔案已儲存為: {save_path}")