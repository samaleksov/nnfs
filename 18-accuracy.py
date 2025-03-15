import numpy as np

softmax_outputs = np.array([[0.7, 0.2, 0.1],
                            [0.5, 0.1, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([0, 1, 1])

predictions = np.argmax(softmax_outputs, axis=1)

if len(class_targets.shape) == 2:
    class_targets = np.argmax(class_targets, axis=1)

accuracy = np.mean(predictions == class_targets)

# How many of our samples are correctly predicted?
print(accuracy)