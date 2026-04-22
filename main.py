import numpy as np  # linear algebra
import struct
from array import array
from os.path import join
import random

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    @staticmethod
    def read_images_labels(images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)


class LinearLayer():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
    
    def forward(self, input_data):
        self.input_data = input_data 
        return np.dot(input_data, self.weights) + self.biases
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input_data.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)
        return input_gradient

class ReLULayer():
    def forward(self, input_data):
        self.input_data = input_data
        return np.maximum(0, input_data)
    
    def backward(self, output_gradient):
        input_gradient = output_gradient.copy()
        input_gradient[self.input_data <= 0] = 0
        return input_gradient
    
class SoftmaxLayer():
    def forward(self, input_data):
        exps = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        self.output_data = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output_data
    
    def backward(self, output_gradient):
        return output_gradient

class CrossEntropyLoss():
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        m = targets.shape[0]
        log_likelihood = -np.log(predictions[range(m), targets] + 1e-9)
        loss = np.sum(log_likelihood) / m
        return loss
    
    def backward(self): 
        m = self.targets.shape[0]
        grad = self.predictions.copy()
        one_hot_targets = np.zeros_like(self.predictions)
        one_hot_targets[np.arange(m), self.targets] = 1
        grad -= one_hot_targets
        grad /= m
        return grad

class NeuralNetwork():
    def __init__(self):
        self.w1 = LinearLayer(784, 64) 
        self.activation1 = ReLULayer()
        self.w2 = LinearLayer(64, 10)
        self.softmax = SoftmaxLayer()
        self.loss_function = CrossEntropyLoss()

    def forward(self, x):
        out = self.w1.forward(x)
        out = self.activation1.forward(out)
        out = self.w2.forward(out)
        out = self.softmax.forward(out)
        return out

    def backward(self, loss_grad, learning_rate):
        grad = self.softmax.backward(loss_grad)
        grad = self.w2.backward(grad, learning_rate)
        grad = self.activation1.backward(grad)
        grad = self.w1.backward(grad, learning_rate)

def main():
    input_path = './dataset'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    nn = NeuralNetwork()
    learning_rate = 0.01
    num_epochs = 20
    batch_size = 64

    x_train_normalized = np.array(x_train) / 255.0
    x_train_flat = x_train_normalized.reshape(x_train_normalized.shape[0], -1)
    y_train_array = np.array(y_train)

    m = x_train_flat.shape[0]

    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        indices = np.arange(m)
        np.random.shuffle(indices)
        x_shuffled = x_train_flat[indices]
        y_shuffled = y_train_array[indices]
        epoch_loss = 0

        for i in range(0, m, batch_size):
            x_batch = x_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            predictions = nn.forward(x_batch)
            loss = nn.loss_function.forward(predictions, y_batch)
            epoch_loss += loss
            
            loss_grad = nn.loss_function.backward()
            nn.backward(loss_grad, learning_rate)

        full_preds = nn.forward(x_train_flat)
        predited_classes = np.argmax(full_preds, axis=1)
        accuracy = np.mean(predited_classes == y_train_array)
        avg_loss = epoch_loss / (m / batch_size)
        
        losses.append(avg_loss)
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    main()