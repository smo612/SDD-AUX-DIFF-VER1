import numpy as np

x_tx = np.load('bridge_tx/x_tx.npy')
y_adc = np.load('bridge/y_adc.npy')

print(f"TX Feature: shape={x_tx.shape}, dtype={x_tx.dtype}")
print(f"ADC Output: shape={y_adc.shape}, dtype={y_adc.dtype}")