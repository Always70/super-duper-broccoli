import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 初始化权重和偏置
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.normal(0, scale, (out_channels, in_channels, kernel_size, kernel_size))
        self.bias = np.zeros(out_channels)
        
    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # 填充输入
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        else:
            x_padded = x
            
        # 初始化输出
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        # 执行卷积
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                x_slice = x_padded[:, :, h_start:h_end, w_start:w_end]
                for k in range(self.out_channels):
                    output[:, k, i, j] = np.sum(x_slice * self.weights[k], axis=(1, 2, 3)) + self.bias[k]
        
        self.last_input = x
        return output
    
    def backward(self, grad_output, learning_rate):
        batch_size, in_channels, height, width = self.last_input.shape
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # 填充输入
        if self.padding > 0:
            x_padded = np.pad(self.last_input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        else:
            x_padded = self.last_input
            
        # 初始化梯度
        grad_weights = np.zeros_like(self.weights)
        grad_bias = np.zeros_like(self.bias)
        grad_input = np.zeros_like(x_padded)
        
        # 计算梯度
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                x_slice = x_padded[:, :, h_start:h_end, w_start:w_end]
                for k in range(self.out_channels):
                    grad_weights[k] += np.sum(x_slice * grad_output[:, k:k+1, i:i+1, j:j+1], axis=0)
                    grad_bias[k] += np.sum(grad_output[:, k, i, j])
                    grad_input[:, :, h_start:h_end, w_start:w_end] += self.weights[k] * grad_output[:, k:k+1, i:i+1, j:j+1]
        
        # 更新参数
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        # 移除填充
        if self.padding > 0:
            grad_input = grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]
            
        return grad_input

class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_height, out_width))
        self.max_indices = np.zeros_like(output, dtype=np.int32)
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                x_slice = x[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.max(x_slice, axis=(2, 3))
                self.max_indices[:, :, i, j] = np.argmax(x_slice.reshape(batch_size, channels, -1), axis=2)
        
        self.last_input = x
        return output
    
    def backward(self, grad_output):
        batch_size, channels, height, width = self.last_input.shape
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        
        grad_input = np.zeros_like(self.last_input)
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                for b in range(batch_size):
                    for c in range(channels):
                        idx = self.max_indices[b, c, i, j]
                        h_idx = idx // self.kernel_size
                        w_idx = idx % self.kernel_size
                        grad_input[b, c, h_start + h_idx, w_start + w_idx] = grad_output[b, c, i, j]
        
        return grad_input

class ReLU:
    def forward(self, x):
        self.last_input = x
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        return grad_output * (self.last_input > 0)

class Flatten:
    def forward(self, x):
        self.last_input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, grad_output):
        return grad_output.reshape(self.last_input_shape)

class Dense:
    def __init__(self, input_size, output_size):
        scale = np.sqrt(2.0 / input_size)
        self.weights = np.random.normal(0, scale, (input_size, output_size))
        self.bias = np.zeros(output_size)
        
    def forward(self, x):
        self.last_input = x
        return np.dot(x, self.weights) + self.bias
    
    def backward(self, grad_output, learning_rate):
        grad_weights = np.dot(self.last_input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0)
        grad_input = np.dot(grad_output, self.weights.T)
        
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        return grad_input

class Softmax:
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output
    
    def backward(self, grad_output):
        batch_size = grad_output.shape[0]
        grad_input = np.zeros_like(grad_output)
        
        for i in range(batch_size):
            jacobian = np.diag(self.output[i]) - np.outer(self.output[i], self.output[i])
            grad_input[i] = np.dot(jacobian, grad_output[i])
        
        return grad_input

class CNN:
    def __init__(self, input_shape, num_classes):
        self.layers = [
            Conv2D(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            
            Conv2D(32, 64, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            
            Flatten(),
            Dense(64 * (input_shape[1]//4) * (input_shape[2]//4), 128),
            ReLU(),
            Dense(128, num_classes),
            Softmax()
        ]
        self.training = True
        
    def train(self):
        """设置模型为训练模式"""
        self.training = True
        
    def eval(self):
        """设置模型为评估模式"""
        self.training = False
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad_output, learning_rate):
        for layer in reversed(self.layers):
            if isinstance(layer, (Conv2D, Dense)):
                grad_output = layer.backward(grad_output, learning_rate)
            else:
                grad_output = layer.backward(grad_output)
        return grad_output
    
    def train_step(self, x, y, learning_rate):
        # 前向传播
        output = self.forward(x)
        
        # 计算损失和梯度
        loss = -np.sum(y * np.log(output + 1e-10)) / x.shape[0]
        grad_output = output - y
        
        # 反向传播
        self.backward(grad_output, learning_rate)
        
        return loss, output
    
    def predict(self, x):
        return self.forward(x) 