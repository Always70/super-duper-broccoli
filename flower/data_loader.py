import numpy as np
import os
from utils import load_image, one_hot_encode

class DataLoader:
    def __init__(self, data_dir, target_size=(32, 32), test_split=0.2):
        self.data_dir = data_dir
        self.target_size = target_size
        self.test_split = test_split
        self.class_names = sorted(os.listdir(data_dir))
        self.num_classes = len(self.class_names)
        
    def load_dataset(self):
        """加载并预处理数据集"""
        X = []
        y = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img_array = load_image(img_path, self.target_size)
                    X.append(img_array)
                    y.append(class_idx)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
        
        X = np.array(X)
        y = np.array(y)
        
        # 随机打乱数据
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # 划分训练集和测试集
        split_idx = int(len(X) * (1 - self.test_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 转换为one-hot编码
        y_train_onehot = one_hot_encode(y_train, self.num_classes)
        y_test_onehot = one_hot_encode(y_test, self.num_classes)
        
        return (X_train, y_train_onehot), (X_test, y_test_onehot), y_test
    
    def get_batch(self, X, y, batch_size):
        """生成批次数据"""
        indices = np.random.permutation(len(X))
        for i in range(0, len(X), batch_size):
            batch_indices = indices[i:i + batch_size]
            yield X[batch_indices], y[batch_indices] 