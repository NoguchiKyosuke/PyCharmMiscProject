import numpy as np

# Load the training data to check dimensions
train_data = np.load('esc_melsp_train_com.npz')
X_train = train_data['x']

print(f'Training data shape: {X_train.shape}')
print(f'Single sample shape: {X_train[0].shape}')
print(f'Height: {X_train.shape[1]}, Width: {X_train.shape[2]}')

# Calculate the output size after conv layers
height, width = X_train.shape[1], X_train.shape[2]
print(f'Original size: {height} x {width}')

# After 4 MaxPool2d layers with kernel_size=2
height_after_pools = height // (2**4)
width_after_pools = width // (2**4)
print(f'Size after 4 pooling layers: {height_after_pools} x {width_after_pools}')

# Final flattened size with 256 channels
flattened_size = 256 * height_after_pools * width_after_pools
print(f'Expected flattened size: {flattened_size}')
