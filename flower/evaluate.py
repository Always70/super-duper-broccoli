import numpy as np
from data_loader import DataLoader
from cnn import CNN
from utils import evaluate_metrics, plot_confusion_matrix
import os


def evaluate(model, data_loader):
    # 加载测试数据
    (_, _), (X_test, y_test), y_test_raw = data_loader.load_dataset()

    # 调整数据格式
    X_test = X_test.transpose(0, 3, 1, 2)

    # 预测
    predictions = []
    true_labels = []

    for i in range(0, len(X_test), 32):
        batch = X_test[i:i + 32]
        output = model.predict(batch)
        pred = np.argmax(output, axis=1)
        predictions.extend(pred)
        true_labels.extend(np.argmax(y_test[i:i + 32], axis=1))

    # 计算评估指标
    metrics = evaluate_metrics(true_labels, predictions)

    # 打印评估结果
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")

    # 绘制混淆矩阵
    plot_confusion_matrix(true_labels, predictions, data_loader.class_names)

    return metrics


if __name__ == '__main__':
    # 加载数据
    data_loader = DataLoader('flowers', target_size=(32, 32))

    # 创建模型
    input_shape = (3, 32, 32)
    num_classes = len(data_loader.class_names)
    model = CNN(input_shape, num_classes)

    # 加载模型权重
    if os.path.exists('model_weights.npy'):
        model.layers = np.load('model_weights.npy', allow_pickle=True)
        print("Model weights loaded successfully")
    else:
        print("No model weights found. Please train the model first.")
        exit(1)

    # 评估模型
    metrics = evaluate(model, data_loader)
