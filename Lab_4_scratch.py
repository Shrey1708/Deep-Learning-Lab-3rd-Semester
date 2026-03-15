"""
CNN Implementation from Scratch
Using only NumPy for mathematical operations
No TensorFlow, PyTorch, or high-level frameworks
"""

import numpy as np
from collections import defaultdict
from PIL import Image
import os
import glob


class ConvLayer:
    """Convolutional Layer - implements 2D convolution from scratch"""
    
    def __init__(self, num_filters, filter_size, input_channels, learning_rate=0.01):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels
        self.learning_rate = learning_rate
        
        # Initialize filters with He initialization
        self.filters = np.random.randn(num_filters, input_channels, filter_size, filter_size) * np.sqrt(2.0 / (filter_size * filter_size * input_channels))
        self.biases = np.zeros((num_filters, 1))
        
        self.cache = {}
    
    def forward(self, X):
        """
        Forward pass of convolution
        X: (batch_size, channels, height, width)
        Returns: (batch_size, num_filters, out_height, out_width)
        """
        batch_size, channels, height, width = X.shape
        
        # Output dimensions
        out_height = height - self.filter_size + 1
        out_width = width - self.filter_size + 1
        
        output = np.zeros((batch_size, self.num_filters, out_height, out_width))
        
        # Apply convolution for each sample and filter
        for b in range(batch_size):
            for f in range(self.num_filters):
                for h in range(out_height):
                    for w in range(out_width):
                        # Extract patch from input
                        patch = X[b, :, h:h+self.filter_size, w:w+self.filter_size]
                        
                        # Element-wise multiplication and sum
                        output[b, f, h, w] = np.sum(patch * self.filters[f]) + self.biases[f, 0]
        
        self.cache['X'] = X
        self.cache['output_shape'] = output.shape
        
        return output
    
    def backward(self, dL_dout):
        """
        Backward pass - compute gradients
        dL_dout: gradient from next layer
        """
        batch_size, channels, height, width = self.cache['X'].shape
        dL_dX = np.zeros_like(self.cache['X'])
        dL_dfilters = np.zeros_like(self.filters)
        dL_dbiases = np.zeros_like(self.biases)
        
        out_height, out_width = dL_dout.shape[2], dL_dout.shape[3]
        
        # Compute gradients for each sample and filter
        for b in range(batch_size):
            for f in range(self.num_filters):
                for h in range(out_height):
                    for w in range(out_width):
                        patch = self.cache['X'][b, :, h:h+self.filter_size, w:w+self.filter_size]
                        dL_dfilters[f] += patch * dL_dout[b, f, h, w]
                        dL_dX[b, :, h:h+self.filter_size, w:w+self.filter_size] += self.filters[f] * dL_dout[b, f, h, w]
                
                dL_dbiases[f] += np.sum(dL_dout[b, f])
        
        # Update weights
        self.filters -= self.learning_rate * dL_dfilters / batch_size
        self.biases -= self.learning_rate * dL_dbiases / batch_size
        
        return dL_dX


class MaxPoolLayer:
    """Max Pooling Layer"""
    
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = {}
    
    def forward(self, X):
        """
        Forward pass of max pooling
        X: (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = X.shape
        
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_height, out_width))
        max_indices = defaultdict(list)
        
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        
                        # Extract patch
                        patch = X[b, c, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size]
                        
                        # Find max value and index
                        output[b, c, h, w] = np.max(patch)
                        max_idx = np.unravel_index(np.argmax(patch), patch.shape)
                        max_indices[(b, c, h, w)] = (h_start + max_idx[0], w_start + max_idx[1])
        
        self.cache['X_shape'] = X.shape
        self.cache['max_indices'] = max_indices
        self.cache['output_shape'] = output.shape
        
        return output
    
    def backward(self, dL_dout):
        """Backward pass for max pooling"""
        dL_dX = np.zeros(self.cache['X_shape'])
        
        batch_size, channels, out_height, out_width = dL_dout.shape
        
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        max_h, max_w = self.cache['max_indices'][(b, c, h, w)]
                        dL_dX[b, c, max_h, max_w] = dL_dout[b, c, h, w]
        
        return dL_dX


class FlattenLayer:
    """Flatten layer to convert feature maps to 1D vector"""
    
    def __init__(self):
        self.cache = {}
    
    def forward(self, X):
        """Flatten: (batch_size, channels, height, width) -> (batch_size, -1)"""
        batch_size = X.shape[0]
        self.cache['X_shape'] = X.shape
        return X.reshape(batch_size, -1)
    
    def backward(self, dL_dout):
        """Restore shape for backward pass"""
        return dL_dout.reshape(self.cache['X_shape'])


class DenseLayer:
    """Fully Connected Layer"""
    
    def __init__(self, input_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Xavier initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        self.biases = np.zeros((1, output_size))
        
        self.cache = {}
    
    def forward(self, X):
        """Forward pass: output = X @ weights + biases"""
        self.cache['X'] = X
        output = np.dot(X, self.weights) + self.biases
        return output
    
    def backward(self, dL_dout):
        """Compute gradients and update weights"""
        batch_size = self.cache['X'].shape[0]
        
        dL_dX = np.dot(dL_dout, self.weights.T)
        dL_dweights = np.dot(self.cache['X'].T, dL_dout) / batch_size
        dL_dbiases = np.sum(dL_dout, axis=0, keepdims=True) / batch_size
        
        # Update weights
        self.weights -= self.learning_rate * dL_dweights
        self.biases -= self.learning_rate * dL_dbiases
        
        return dL_dX


class ReLU:
    """ReLU Activation Function"""
    
    def __init__(self):
        self.cache = {}
    
    def forward(self, X):
        """ReLU: max(0, X)"""
        self.cache['X'] = X
        return np.maximum(0, X)
    
    def backward(self, dL_dout):
        """Gradient of ReLU: 1 if X > 0, else 0"""
        X = self.cache['X']
        return dL_dout * (X > 0)


class Softmax:
    """Softmax Activation (for output layer)"""
    
    def __init__(self):
        self.cache = {}
    
    def forward(self, X):
        """Softmax normalization"""
        # Subtract max for numerical stability
        X_shifted = X - np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(X_shifted)
        softmax = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        self.cache['softmax'] = softmax
        return softmax
    
    def backward(self, dL_dout):
        """Gradient of softmax"""
        softmax = self.cache['softmax']
        batch_size = softmax.shape[0]
        
        dL_dX = np.zeros_like(dL_dout)
        for i in range(batch_size):
            s = softmax[i]
            dL_dX[i] = s * (dL_dout[i] - np.dot(dL_dout[i], s))
        
        return dL_dX


class CrossEntropyLoss:
    """Cross Entropy Loss Function"""
    
    @staticmethod
    def forward(predictions, labels):
        """
        predictions: (batch_size, num_classes) - softmax output
        labels: (batch_size,) - class indices
        """
        batch_size = predictions.shape[0]
        log_predictions = np.log(predictions + 1e-10)  # Add small value to avoid log(0)
        loss = -np.mean(log_predictions[np.arange(batch_size), labels])
        return loss
    
    @staticmethod
    def backward(predictions, labels):
        """Gradient of cross entropy"""
        batch_size = predictions.shape[0]
        dL_dout = predictions.copy()
        dL_dout[np.arange(batch_size), labels] -= 1
        dL_dout /= batch_size
        return dL_dout


class CNN:
    """CNN with multiple Conv blocks for better feature extraction"""
    
    def __init__(self, num_classes=2, learning_rate=0.01):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # Build layers - deeper network
        self.conv1 = ConvLayer(num_filters=32, filter_size=3, input_channels=3, learning_rate=learning_rate)
        self.relu1 = ReLU()
        self.pool1 = MaxPoolLayer(pool_size=2, stride=2)
        
        self.conv2 = ConvLayer(num_filters=64, filter_size=3, input_channels=32, learning_rate=learning_rate)
        self.relu2 = ReLU()
        self.pool2 = MaxPoolLayer(pool_size=2, stride=2)
        
        self.conv3 = ConvLayer(num_filters=128, filter_size=3, input_channels=64, learning_rate=learning_rate)
        self.relu3 = ReLU()
        self.pool3 = MaxPoolLayer(pool_size=2, stride=2)
        
        self.flatten = FlattenLayer()
        self.dense1 = DenseLayer(input_size=128*2*2, output_size=256, learning_rate=learning_rate)
        self.relu4 = ReLU()
        self.dense2 = DenseLayer(input_size=256, output_size=128, learning_rate=learning_rate)
        self.relu5 = ReLU()
        self.dense3 = DenseLayer(input_size=128, output_size=num_classes, learning_rate=learning_rate)
        self.softmax = Softmax()
    
    def forward(self, X):
        """Forward pass through entire network"""
        # Conv block 1
        x = self.conv1.forward(X)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        
        # Conv block 2
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        
        # Conv block 3
        x = self.conv3.forward(x)
        x = self.relu3.forward(x)
        x = self.pool3.forward(x)
        
        # Dense layers
        x = self.flatten.forward(x)
        x = self.dense1.forward(x)
        x = self.relu4.forward(x)
        x = self.dense2.forward(x)
        x = self.relu5.forward(x)
        x = self.dense3.forward(x)
        x = self.softmax.forward(x)
        
        return x
    
    def backward(self, dL_dout):
        """Backward pass through entire network"""
        # Reverse order of forward pass
        dL_dout = self.softmax.backward(dL_dout)
        dL_dout = self.dense3.backward(dL_dout)
        dL_dout = self.relu5.backward(dL_dout)
        dL_dout = self.dense2.backward(dL_dout)
        dL_dout = self.relu4.backward(dL_dout)
        dL_dout = self.dense1.backward(dL_dout)
        dL_dout = self.flatten.backward(dL_dout)
        dL_dout = self.pool3.backward(dL_dout)
        dL_dout = self.relu3.backward(dL_dout)
        dL_dout = self.conv3.backward(dL_dout)
        dL_dout = self.pool2.backward(dL_dout)
        dL_dout = self.relu2.backward(dL_dout)
        dL_dout = self.conv2.backward(dL_dout)
        dL_dout = self.pool1.backward(dL_dout)
        dL_dout = self.relu1.backward(dL_dout)
        dL_dout = self.conv1.backward(dL_dout)
    
    def train_step(self, X_batch, y_batch):
        """Single training step"""
        # Forward pass
        predictions = self.forward(X_batch)
        
        # Compute loss
        loss = CrossEntropyLoss.forward(predictions, y_batch)
        
        # Backward pass
        dL_dout = CrossEntropyLoss.backward(predictions, y_batch)
        self.backward(dL_dout)
        
        return loss
    
    def predict(self, X):
        """Get model predictions"""
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)


# Example usage
if __name__ == "__main__":
    # Load real data from dataset
    def load_dataset(image_dir, label_dir, img_size=32):
        images = []
        labels = []
        
        # Get all image files
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        
        for img_path in image_files:
            # Get corresponding label file
            basename = os.path.basename(img_path).replace('.jpg', '.txt')
            label_path = os.path.join(label_dir, basename)
            
            if not os.path.exists(label_path):
                continue
            
            try:
                # Load image
                img = Image.open(img_path).convert('RGB')
                img = img.resize((img_size, img_size))
                img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                img_array = np.transpose(img_array, (2, 0, 1))  # (H, W, C) -> (C, H, W)
                images.append(img_array)
                
                # Get label (first value in the label file is class)
                with open(label_path, 'r') as f:
                    label_data = f.readline().strip().split()
                    label = int(float(label_data[0]))  # Class ID
                    labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        return np.array(images), np.array(labels)
    
    print("Loading dataset...")
    train_images, train_labels = load_dataset(
        r"c:\Users\shrey\Downloads\signature\images\train",
        r"c:\Users\shrey\Downloads\signature\labels\train"
    )
    
    val_images, val_labels = load_dataset(
        r"c:\Users\shrey\Downloads\signature\images\val",
        r"c:\Users\shrey\Downloads\signature\labels\val"
    )
    
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    print(f"Unique classes: {np.unique(train_labels)}")
    
    # Determine number of classes
    num_classes = max(int(np.max(train_labels)) + 1, 2)
    print(f"Number of classes: {num_classes}")
    
    if len(train_images) == 0:
        print("No training data found! Using random data...")
        X_train = np.random.randn(100, 3, 32, 32) / 255.0
        y_train = np.random.randint(0, num_classes, 100)
        X_val = np.random.randn(20, 3, 32, 32) / 255.0
        y_val = np.random.randint(0, num_classes, 20)
    else:
        X_train = train_images
        y_train = train_labels
        X_val = val_images if len(val_images) > 0 else train_images[:20]
        y_val = val_labels if len(val_labels) > 0 else train_labels[:20]
    
    # Create and train model with improved hyperparameters
    model = CNN(num_classes=num_classes, learning_rate=0.1)
    
    print("\nTraining CNN from scratch...")
    batch_size = 16
    epochs = 20
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = len(X_train) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            loss = model.train_step(X_batch, y_batch)
            total_loss += loss
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Validation
        val_predictions = model.predict(X_val)
        val_accuracy = np.mean(val_predictions == y_val)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    # Final validation
    predictions = model.predict(X_val)
    accuracy = np.mean(predictions == y_val)
    print(f"\nFinal Validation Accuracy: {accuracy:.4f}")
