import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm  # 导入 tqdm
import random
import os
import time

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

import logging

# 配置日志记录，输出到控制台和 run_gradual_new.txt 文件
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s: %(message)s',
                    handlers=[
                        logging.FileHandler("gradual.txt")

                    ])

# 配置类，将所有超参数集中管理
class Config:
    # 随机种子
    RANDOM_SEED = 42
    
    # 数据路径
    TRAIN_DATA_PATH = "C:/Users/hsuns/Desktop/concept+drift/data/kddcup.data_10_percent_t.csv"
    TEST_DATA_PATH = "C:/Users/hsuns/Desktop/concept+drift/data/df_drifted_top25_gradual.csv"
    
    # DNN参数
    DNN_EPOCHS = 1
    DNN_LR = 0.001
    DNN_BATCH_SIZE = 64
    
    # 统计漂移检测器参数
    STAT_WINDOW_SIZE = 100
    STAT_THRESHOLD = 3.0
    
    # 自动编码器漂移检测器参数
    AE_LONG_WINDOW_SIZE = 50
    AE_SHORT_WINDOW_SIZE = 1
    AE_SIMILARITY_THRESHOLD = 0.999
    AE_DRIFT_COUNT_THRESHOLD = 10
    AE_SIM_THRESHOLD = 0.5
    AE_THRESHOLD_SIM1 = 0.85
    AE_THRESHOLD_SIM2 = 0.85
    AE_THRESHOLD_DIFF = 0.1
    AE_M1_EPOCHS = 20
    AE_M2_EPOCHS = 20
    AE_BATCH_SIZE = 32
    
    # 重新训练DNN参数
    RETRAIN_DNN_EPOCHS = 50
    RETRAIN_DNN_BATCH_SIZE = 64
    
    # 其他参数
    LOG_FILE = "gradual.txt"
    TEST_EVALUATION_RESULTS_FILE = 'test_evaluation_results.txt'
    EVALUATION_RESULTS_FILE = 'evaluation_results.txt'

# 设置随机种子以确保结果的可复现性
def set_seed(seed=Config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)

def preprocess_data(df):
    """预处理输入的DataFrame数据"""
    df_copy = df.copy()

    # 如果存在'class'列，则重命名为'label'
    if 'class' in df_copy.columns:
        df_copy = df_copy.rename(columns={'class': 'label'})
    else:
        raise KeyError("DataFrame中不存在 'class' 列，请检查列名是否正确。")

    # 识别数值型列，排除标签列，并且每列唯一值超过2个
    number_col = df_copy.select_dtypes(include=['float64', 'int64']).columns
    number_col = [col for col in number_col if col != 'label' and df_copy[col].nunique() > 2]

    # 对数值特征进行标准化
    scaler = StandardScaler()
    if number_col:
        df_copy[number_col] = scaler.fit_transform(df_copy[number_col])

    # 进行归一化
    min_max_scaler = MinMaxScaler()
    if number_col:
        df_copy[number_col] = min_max_scaler.fit_transform(df_copy[number_col])

    # 对标签进行编码
    if 'label' in df_copy.columns:
        label_encoder = LabelEncoder()
        df_copy['label'] = label_encoder.fit_transform(df_copy['label'])

    # 确保所有数据都是数值类型，并处理缺失值
    df_copy = df_copy.apply(pd.to_numeric, errors='coerce')
    if df_copy.isnull().values.any():
        df_copy = df_copy.fillna(0)

    # 验证所有列是否为数值类型
    for col in df_copy.columns:
        if not pd.api.types.is_numeric_dtype(df_copy[col]):
            logging.warning(f"列 '{col}' 不是数值类型，尝试转换为数值类型。")
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)

    # 最后，确认标签只包含 0 和 1
    unique_labels = df_copy['label'].unique()
    logging.info(f"唯一标签值: {unique_labels}")
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"标签值不在预期范围内 [0, 1]. 当前标签值: {unique_labels}")

    return df_copy

# 自定义数据集类
class LoadData(Dataset):
    def __init__(self, X, y):
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = torch.tensor(self.X.iloc[index].values, dtype=torch.float32)
        y = torch.tensor(int(self.y.iloc[index]), dtype=torch.long)
        return X, y

# 深度神经网络（DNN）模型定义
class DNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DNN, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),  # 增加神经元数量以适应更高维度
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, X):
        X = self.flatten(X)
        logits = self.network(X)
        return logits

# 训练函数
def train_model(model, optimizer, loss_fn, epochs, train_dataloader, device, input_dim):
    model.train()  # 设置模型为训练模式
    start_time = time.time()

    with tqdm(total=epochs, desc="训练进度", unit="epoch") as pbar:
        for epoch in range(epochs):
            epoch_loss = 0.0
            for X, y in train_dataloader:
                X, y = X.to(device), y.to(device)
                X = X.view(X.size(0), input_dim)  # 移除不必要的维度
                y_pred = model(X)
                loss = loss_fn(y_pred, y)

                optimizer.zero_grad()  # 梯度清零
                loss.backward()        # 反向传播
                optimizer.step()       # 更新参数

                epoch_loss += loss.item()

            # 在训练集上评估模型
            train_acc, train_precision, train_recall, train_f1, _ = evaluate_model(
                model, train_dataloader, loss_fn, device, input_dim
            )

            # 将训练指标添加到 tqdm 描述中
            pbar.set_postfix({
                "准确率": f"{train_acc:.4f}",
                "精度": f"{train_precision:.4f}",
                "召回率": f"{train_recall:.4f}",
                "F1": f"{train_f1:.4f}"
            })
            pbar.update(1)

    train_time = time.time() - start_time
    logging.info(f"DNN模型训练完成，耗时: {train_time:.2f}秒")

    return train_time

# 评估函数
def evaluate_model(model, dataloader, loss_fn, device, input_dim):
    model.eval()  # 设置模型为评估模式
    y_true = []
    y_pred_list = []
    loss_sum = 0

    with torch.no_grad():  # 禁用梯度计算
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.view(X.size(0), input_dim)  # 移除不必要的维度
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss_sum += loss.item()

            _, preds = torch.max(outputs, 1)
            y_pred_list.extend(preds.cpu().numpy())
            y_true.extend(y.cpu().numpy())

    y_pred = np.array(y_pred_list)
    y_true = np.array(y_true)

    # 计算性能指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1, loss_sum

# 统计方法概念漂移检测器类（3西格玛原则）
class ConceptDriftDetectorStat:
    def __init__(self, window_size=Config.STAT_WINDOW_SIZE, threshold=Config.STAT_THRESHOLD):
        """
        初始化漂移检测器

        :param window_size: 用于计算均值、方差等的滑动窗口大小
        :param threshold: Z-score阈值，用于检测漂移（3西格玛原则）
        """
        self.window_size = window_size  # 窗口大小
        self.threshold = threshold  # Z-score阈值
        self.history_data = []  # 存储历史数据

    def _mean_and_std(self, data):
        """计算数据的均值和标准差"""
        mean = np.mean(data)
        std_dev = np.std(data)
        return mean, std_dev

    def _z_score(self, new_data, mean, std_dev):
        """计算Z-score并与阈值比较"""
        if std_dev == 0:
            return 0.0
        return (new_data - mean) / std_dev

    def detect_drift(self, new_data):
        """
        检测是否发生概念漂移

        :param new_data: 新数据点（单个值或单个特征的数组）
        :return: 是否发生漂移
        """
        if len(self.history_data) < self.window_size:
            # 如果历史数据不够填满窗口，先将新数据添加到历史数据中
            self.history_data.append(new_data)
            return False

        # 当历史数据已满时，滑动窗口将删除最旧的数据
        self.history_data.pop(0)
        self.history_data.append(new_data)

        # 计算当前窗口数据的均值和标准差
        mean, std_dev = self._mean_and_std(self.history_data)

        # 计算新数据点的Z-score
        z_score = self._z_score(new_data, mean, std_dev)

        # 如果Z-score超过设定的阈值，认为发生了概念漂移
        if abs(z_score) > self.threshold:
            logging.info(f"概念漂移检测 (Z-score): 新数据的Z-score = {z_score:.2f} 超出阈值")
            return True

        return False

# 概念漂移检测器类（使用自动编码器）
class ConceptDriftDetectorAutoEncoder: 
    def __init__(self, config: Config, input_dim: int):  
        """ 
        初始化概念漂移检测器 

        Args: 
            config (Config): 配置对象
            input_dim (int): 输入数据的维度
        """ 
        self.input_dim = input_dim  # 正确设置输入维度
        self.long_window_size = config.AE_LONG_WINDOW_SIZE
        self.short_window_size = config.AE_SHORT_WINDOW_SIZE
        self.similarity_threshold = config.AE_SIMILARITY_THRESHOLD
        self.drift_count_threshold = config.AE_DRIFT_COUNT_THRESHOLD
        self.sim_threshold = config.AE_SIM_THRESHOLD  # 新增相似度阈值

        # 新增的阈值
        self.threshold_sim1 = config.AE_THRESHOLD_SIM1
        self.threshold_sim2 = config.AE_THRESHOLD_SIM2
        self.threshold_diff = config.AE_THRESHOLD_DIFF

        # 数据集 
        self.D1 = []  # 正常数据 
        self.D2 = []  # 漂移数据 

        # 初始化自动编码器（使用Keras） 
        self.M1 = self._build_autoencoder()  # 正常数据自动编码器 
        self.M2 = self._build_autoencoder()  # 漂移数据自动编码器 

        # 缓冲区 
        self.window_buffer = [] 

        # 漂移检测相关属性 
        self.consecutive_drift_count = 0 
        self.drift_positions = [] 

    def _build_autoencoder(self):
        """构建自动编码器"""
        input_layer = Input(shape=(self.input_dim,))

        # 编码器
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dense(32, activation='relu')(encoded)

        # 解码器
        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(self.input_dim, activation='linear')(decoded)

        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse')

        return autoencoder

    def cosine_similarity(self, a, b):
        """计算两个向量之间的余弦相似度"""
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def check_long_window(self, window_data, scheme='scheme2'):
        """
        检查长窗口中的数据是否可能存在漂移

        Args:
            window_data (list of np.array): 窗口中的数据
            scheme (str): 'scheme1' 或 'scheme2'

        Returns:
            bool: 是否检测到潜在漂移
            float: 平均相似度
        """
        similarities = []
        window_array = np.array(window_data)

        # 验证数据类型
        if not np.issubdtype(window_array.dtype, np.floating):
            raise ValueError(f"window_array 包含非浮点数类型: {window_array.dtype}")

        if scheme == 'scheme1':
            # scheme1: compare with M2, drift if similarity > threshold
            reconstructed = self.M2.predict(window_array, verbose=0)
            for orig, recon in zip(window_array, reconstructed):
                sim = self.cosine_similarity(orig, recon)
                similarities.append(sim)
            avg_similarity = np.mean(similarities)
            drift_detected = avg_similarity > self.similarity_threshold  # 相似度高于阈值表示可能漂移
        elif scheme == 'scheme2':
            # scheme2: compare with M1, drift if similarity < threshold
            reconstructed = self.M1.predict(window_array, verbose=0)
            similarities = [self.cosine_similarity(orig, recon) for orig, recon in zip(window_array, reconstructed)]
            avg_similarity = np.mean(similarities)
            drift_detected = avg_similarity < self.similarity_threshold  # 相似度低于阈值表示可能漂移
        else:
            raise ValueError(f"未知的漂移检测方案: {scheme}")

        if drift_detected:
            logging.info("\n=== 长窗口漂移检测 ===")
            if scheme == 'scheme1':
                logging.info(f"窗口平均相似度: {avg_similarity:.4f} > 阈值 {self.similarity_threshold}")
            else:
                logging.info(f"窗口平均相似度: {avg_similarity:.4f} < 阈值 {self.similarity_threshold}")
            logging.info(f"是否检测到潜在漂移: {drift_detected}")
            logging.info("=" * 40)
        else:
            if scheme == 'scheme1':
                logging.info(f"窗口平均相似度: {avg_similarity:.4f} <= 阈值 {self.similarity_threshold}, 无漂移")
            else:
                logging.info(f"窗口平均相似度: {avg_similarity:.4f} >= 阈值 {self.similarity_threshold}, 无漂移")

        return drift_detected, avg_similarity

    def check_short_window(self, sample, position):
        """ 
        检查单个样本是否为漂移点 

        Args: 
            sample (np.array): 待检测样本 
            position (int): 样本在窗口中的位置 

        Returns: 
            bool: 是否为漂移点 
            tuple: (sim1, sim2)
        """ 
        sample = np.array([sample]) 

        # 使用两个自动编码器进行重构 
        O1 = self.M1.predict(sample, verbose=0) 
        O2 = self.M2.predict(sample, verbose=0) 

        # 计算相似度 
        sim1 = self.cosine_similarity(sample[0], O1[0]) 
        sim2 = self.cosine_similarity(sample[0], O2[0]) 

        # 计算 sim_diff
        sim_diff = abs(sim1 - sim2)

        # 判断是否为漂移点
        drift_detected = (sim1 < self.threshold_sim1) or (sim_diff > self.threshold_diff)

        logging.info(f"位置 {position} - sim1: {sim1:.4f}, sim2: {sim2:.4f}, |sim1 - sim2|: {sim_diff:.4f}, 漂移: {drift_detected}") 

        if drift_detected: 
            self.consecutive_drift_count += 1 
            self.drift_positions.append(position) 
        else: 
            self.consecutive_drift_count = 0  # 重置连续漂移计数 

        return drift_detected, (sim1, sim2)

    def retrain_models(self, drift_data, drift_labels, normal_data, normal_labels, dnn_model=None, dnn_optimizer=None, dnn_loss_fn=None, device=None, dnn_epochs=Config.RETRAIN_DNN_EPOCHS):
        """
        重新训练自动编码器和DNN模型

        Args:
            drift_data (list of np.array): 漂移数据
            drift_labels (list of int): 漂移数据的标签
            normal_data (list of np.array): 正常数据
            normal_labels (list of int): 正常数据的标签
            dnn_model (torch.nn.Module): DNN模型
            dnn_optimizer (torch.optim.Optimizer): DNN优化器
            dnn_loss_fn (torch.nn.Module): DNN损失函数
            device (torch.device): 训练设备
            dnn_epochs (int): DNN重新训练的epoch数
        """
        # 重新训练自动编码器
        if normal_data:
            self.D1.extend(normal_data)
            logging.info(f"使用 {len(normal_data)} 个正常样本训练M1")
            self.M1.fit(np.array(normal_data), np.array(normal_data),
                       epochs=Config.AE_M1_EPOCHS, batch_size=Config.AE_BATCH_SIZE, verbose=0)

        if drift_data:
            self.D2.extend(drift_data)
            logging.info(f"使用 {len(drift_data)} 个漂移样本训练M2")
            self.M2.fit(np.array(drift_data), np.array(drift_data),
                       epochs=Config.AE_M2_EPOCHS, batch_size=Config.AE_BATCH_SIZE, verbose=0)

        # 如果提供了DNN模型，则使用漂移数据重新训练DNN模型
        if dnn_model and dnn_optimizer and dnn_loss_fn and device:
            if drift_data:
                logging.info("使用漂移数据重新训练DNN模型...")
                # 将漂移数据及其标签转换为DataLoader
                drift_dataset = TensorDataset(torch.tensor(np.array(drift_data)).float(),
                                             torch.tensor(np.array(drift_labels)).long())
                drift_dataloader = DataLoader(drift_dataset, batch_size=Config.RETRAIN_DNN_BATCH_SIZE, shuffle=True)

                # 使用 tqdm 添加进度条
                for epoch in tqdm(range(dnn_epochs), desc="DNN重新训练进度", unit="epoch"):
                    dnn_model.train()
                    for X, y in tqdm(drift_dataloader, desc=f"Epoch {epoch + 1}/{dnn_epochs}", leave=False, unit="batch"):
                        X, y = X.to(device), y.to(device)

                        dnn_model.zero_grad()
                        outputs = dnn_model(X)
                        loss = dnn_loss_fn(outputs, y)
                        loss.backward()
                        dnn_optimizer.step()

            logging.info("DNN模型重新训练完成。")

    def process_sample(self, sample, label, dnn_model=None, dnn_optimizer=None, dnn_loss_fn=None, device=None, dnn_epochs=Config.RETRAIN_DNN_EPOCHS, scheme_long='scheme2'):
        """
        处理单个样本进行漂移检测

        Args:
            sample (np.array): 待检测样本
            label (int): 样本的真实标签
            dnn_model (torch.nn.Module): DNN模型
            dnn_optimizer (torch.optim.Optimizer): DNN优化器
            dnn_loss_fn (torch.nn.Module): DNN损失函数
            device (torch.device): 训练设备
            dnn_epochs (int): DNN重新训练的epoch数
            scheme_long (str): 长窗口漂移检测方案（'scheme1' 或 'scheme2'）

        Returns:
            tuple: (is_drift, drift_info)
        """
        self.window_buffer.append((sample, label))

        if len(self.window_buffer) >= self.long_window_size:
            # 检查长窗口是否存在漂移
            window_data = [data for data, _ in self.window_buffer]
            potential_drift, _ = self.check_long_window(window_data, scheme=scheme_long)

            if potential_drift:
                drift_data = [] 
                drift_labels = []
                normal_data = [] 
                normal_labels = []
                self.drift_positions = [] 

                # 进行短窗口检测
                for i in range(self.long_window_size): 
                    sample_i, label_i = self.window_buffer[i]
                    is_drift, similarities = self.check_short_window(sample_i, i) 
                    if is_drift: 
                        drift_data.append(sample_i) 
                        drift_labels.append(label_i)
                    else: 
                        normal_data.append(sample_i)
                        normal_labels.append(label_i)

                    # 当连续漂移点达到阈值时，触发重训练 
                    if self.consecutive_drift_count >= self.drift_count_threshold: 
                        logging.info(f"检测到渐进漂移，连续漂移点数量: {self.consecutive_drift_count}") 

                        # 重新训练自动编码器和DNN模型 
                        self.retrain_models(drift_data, drift_labels, normal_data, normal_labels, dnn_model, dnn_optimizer, dnn_loss_fn, device, dnn_epochs) 

                        # 整理漂移信息 
                        drift_info = { 
                            'drift_count': len(drift_data), 
                            'normal_count': len(normal_data), 
                            'drift_positions': self.drift_positions.copy(), 
                            # 'similarities': similarities  # 可选，包含相似度信息 
                        } 

                        # 重置缓冲区和计数器 
                        self.window_buffer = [] 
                        self.consecutive_drift_count = 0 
                        self.drift_positions = [] 

                        return True, drift_info 

            # 移除最老的样本以保持窗口大小 
            self.window_buffer.pop(0) 

        return False, None

# 主函数
def main():
    # 设置随机种子
    set_seed(Config.RANDOM_SEED)

    # 1. 加载 D1 数据用于训练
    print("加载训练数据 (D1)...")
    df_train = pd.read_csv(Config.TRAIN_DATA_PATH)

    # 预处理 D1 数据
    print("预处理训练数据...")
    df_train_processed = preprocess_data(df_train)
    
    X_train = df_train_processed.drop(columns=['label'])
    y_train = df_train_processed['label']
    
    # 设置输入维度
    input_dim = X_train.shape[1]
    output_dim = y_train.nunique()

    # 打印唯一标签值以进行调试
    print(f"训练集标签分布:\n{y_train.value_counts()}")

    # 创建训练集的 TensorDataset 和 DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train.values).float(),
                                  torch.tensor(y_train.values).long())
    train_dataloader = DataLoader(train_dataset, batch_size=Config.DNN_BATCH_SIZE, shuffle=True)

    # 2. 训练DNN模型
    # 配置训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化DNN模型
    print(f"输入维度: {input_dim}, 输出维度 (类别数): {output_dim}")
    model = DNN(input_dim=input_dim, output_dim=output_dim).to(device)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=Config.DNN_LR)
    loss_fn = nn.CrossEntropyLoss()

    # 训练DNN模型
    print("开始训练DNN模型...")
    train_time = train_model(
        model, optimizer, loss_fn, Config.DNN_EPOCHS, train_dataloader, device, input_dim
    )

    # 3. 使用统计方法进行概念漂移检测
    print("\n初始化统计概念漂移检测器...")
    detector_stat = ConceptDriftDetectorStat(window_size=Config.STAT_WINDOW_SIZE, threshold=Config.STAT_THRESHOLD)

    # 计算D1数据集的预测误差并进行漂移检测
    print("开始使用统计方法进行概念漂移检测...")
    errors = []
    model.eval()
    with torch.no_grad():  # 禁用梯度计算
        for X, y in tqdm(train_dataloader, desc="计算预测误差", unit="batch"):
            X = X.to(device)
            y = y.to(device)
            X = X.view(X.size(0), input_dim)
            outputs = model(X)
            loss = nn.CrossEntropyLoss(reduction='none')(outputs, y)  # 计算每个样本的损失
            batch_errors = loss.cpu().numpy()
            errors.extend(batch_errors)

    # 使用统计方法检测漂移
    drift_indices = []
    for i, error in enumerate(tqdm(errors, desc="漂移检测进度", unit="样本")):
        drift = detector_stat.detect_drift(error)
        if drift:
            drift_indices.append(i)

    logging.info("\n统计概念漂移检测结果:")
    logging.info(f"总样本数: {len(errors)}")
    logging.info(f"检测到的漂移次数: {len(drift_indices)}")
    if drift_indices:
        logging.info(f"漂移发生的位置: {drift_indices}")

    # 根据漂移检测结果将D1数据集分为非漂移数据和漂移数据
    logging.info("\n将D1数据集分为非漂移数据和漂移数据...")
    non_drift_indices = list(set(range(len(df_train_processed))) - set(drift_indices))
    non_drift_data = df_train_processed.iloc[non_drift_indices]
    drift_data = df_train_processed.iloc[drift_indices]

    logging.info(f"非漂移数据样本数: {len(non_drift_data)}")
    logging.info(f"漂移数据样本数: {len(drift_data)}")

    # 4. 训练自动编码器 M1 和 M2
    logging.info("\n训练自动编码器 M1 和 M2...")
    if len(non_drift_data) > 0:
        X_non_drift = non_drift_data.drop(columns=['label']).values
        y_non_drift = non_drift_data['label'].values
    else:
        X_non_drift = np.array([])
        y_non_drift = np.array([])

    if len(drift_data) > 0:
        X_drift = drift_data.drop(columns=['label']).values
        y_drift = drift_data['label'].values
    else:
        X_drift = np.array([])
        y_drift = np.array([])

    # 初始化自动编码器漂移检测器
    logging.info("初始化自动编码器漂移检测器...") 
    detector_autoencoder = ConceptDriftDetectorAutoEncoder(config=Config, input_dim=input_dim) 

    # 使用非漂移数据训练M1和漂移数据训练M2
    logging.info("训练自动编码器 M1 和 M2...")
    with tqdm(total=1, desc="训练M1和M2", unit="batch") as pbar:
        if len(X_non_drift) > 0:
            logging.info("使用非漂移数据训练M1...")
            detector_autoencoder.M1.fit(X_non_drift, X_non_drift, 
                                         epochs=Config.AE_M1_EPOCHS, 
                                         batch_size=Config.AE_BATCH_SIZE, verbose=0)
        else:
            logging.info("没有非漂移数据用于训练M1。")

        if len(X_drift) > 0:
            logging.info("使用漂移数据训练M2...")
            detector_autoencoder.M2.fit(X_drift, X_drift, 
                                         epochs=Config.AE_M2_EPOCHS, 
                                         batch_size=Config.AE_BATCH_SIZE, verbose=0)
        else:
            logging.info("没有漂移数据用于训练M2。")
        pbar.update(1)

    # 5. 加载 D2 数据用于测试
    logging.info("\n加载测试数据 (D2)...")
    df_test = pd.read_csv(Config.TEST_DATA_PATH) 
    # 预处理 D2 数据
    logging.info("预处理测试数据...")
    df_test_processed = preprocess_data(df_test)
    X_test = df_test_processed.drop(columns=['label'])
    y_test = df_test_processed['label']

    # 打印标签分布以进行调试
    logging.info(f"测试集标签分布:\n{y_test.value_counts()}")

    # 创建测试集的 TensorDataset 和 DataLoader
    test_dataset = TensorDataset(torch.tensor(X_test.values).float(),
                                 torch.tensor(y_test.values).long())
    test_dataloader = DataLoader(test_dataset, batch_size=Config.DNN_BATCH_SIZE, shuffle=False)

    # 在测试集上评估DNN模型
    logging.info("在测试集上评估DNN模型...")
    test_acc, test_precision, test_recall, test_f1, test_loss_sum = evaluate_model(
        model, test_dataloader, loss_fn, device, input_dim
    )
    logging.info(f"测试集评估结果:")
    logging.info(f"准确率: {test_acc:.4f}")
    logging.info(f"精度: {test_precision:.4f}")
    logging.info(f"召回率: {test_recall:.4f}")
    logging.info(f"F1分数: {test_f1:.4f}")

    # 保存测试集评估结果
    with open(Config.TEST_EVALUATION_RESULTS_FILE, 'w', encoding='utf-8') as file:
        file.write("测试集评估结果:\n")
        file.write(f"准确率: {test_acc:.4f}\n")
        file.write(f"精度: {test_precision:.4f}\n")
        file.write(f"召回率: {test_recall:.4f}\n")
        file.write(f"F1分数: {test_f1:.4f}\n")

    # 6. 使用D2进行概念漂移检测
    logging.info("\n开始概念漂移检测 on D2...")
    drift_detections = []
    # 长窗口使用 'scheme2'（根据您的方法说明）
    scheme_long = 'scheme2'

    # 使用D2数据进行漂移检测
    logging.info(f"\n使用 scheme2 进行长窗口漂移检测，使用 scheme2 进行短窗口漂移检测...")
    for i, (sample, label) in enumerate(tqdm(zip(X_test.values, y_test.values), desc="漂移检测进度", total=len(X_test))):
        # 处理样本进行漂移检测
        is_drift, drift_info = detector_autoencoder.process_sample( 
            sample,  
            label,  # 传递标签
            dnn_model=model,  
            dnn_optimizer=optimizer,  
            dnn_loss_fn=loss_fn,  
            device=device,  
            dnn_epochs=Config.RETRAIN_DNN_EPOCHS, 
            scheme_long=scheme_long 
        )
        if is_drift:
            logging.info(f"\n在位置 {i} 检测到漂移:")
            logging.info(f"漂移点数量: {drift_info['drift_count']}")
            logging.info(f"漂移点位置: {drift_info['drift_positions']}")
            drift_detections.append((i, drift_info))

    # 输出漂移检测的统计信息
    logging.info("\n漂移检测统计:")
    logging.info(f"总样本数: {len(X_test)}")
    logging.info(f"检测到的漂移次数: {len(drift_detections)}")
    if drift_detections:
        drift_positions = [pos for pos, _ in drift_detections]
        logging.info(f"漂移发生的位置: {drift_positions}")

    # 7. 再次评估D2数据集
    logging.info("\n进行分类检测 after Drift Detection...")
    # 创建 Dataset 和 DataLoader
    classification_dataset = TensorDataset(torch.tensor(X_test.values).float(),
                                          torch.tensor(y_test.values).long())
    classification_dataloader = DataLoader(classification_dataset, batch_size=Config.DNN_BATCH_SIZE, shuffle=False)

    # 评估模型在整个数据集上的表现
    all_preds = []
    all_labels = []

    logging.info("开始进行分类检测...")
    with tqdm(classification_dataloader, desc="分类检测进度", unit="batch") as pbar:
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            X_batch = X_batch.view(X_batch.size(0), input_dim)
            with torch.no_grad():
                outputs = model(X_batch)
                _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    # 计算性能指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    logging.info("\n整个数据集的分类评估结果 after Drift Detection:")
    logging.info(f"准确率: {accuracy:.4f}")
    logging.info(f"精度: {precision:.4f}")
    logging.info(f"召回率: {recall:.4f}")
    logging.info(f"F1分数: {f1:.4f}")

    # 保存分类评估结果
    with open(Config.EVALUATION_RESULTS_FILE, 'w', encoding='utf-8') as file:
        file.write("整个数据集的分类评估结果 after Drift Detection:\n")
        file.write(f"准确率: {accuracy:.4f}\n")
        file.write(f"精度: {precision:.4f}\n")
        file.write(f"召回率: {recall:.4f}\n")
        file.write(f"F1分数: {f1:.4f}\n")

if __name__ == "__main__":
    main()
