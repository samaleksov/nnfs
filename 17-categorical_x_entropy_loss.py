import numpy as np

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(class_targets.shape) == 1:
            correct_confidences = softmax_outputs[
                range(len(softmax_outputs)),
                class_targets
            ]
        elif len(class_targets.shape) == 2:
            correct_confidences = np.sum(softmax_outputs*class_targets, axis=1)

        neg_log_likelihoods = -np.log(correct_confidences)
        return neg_log_likelihoods


softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

# What is the output if we make better or worse predictions?
#softmax_outputs = np.array([[1.0, 0.0, 0.0],
#                            [0.0, 1.0, 0],
#                            [0.0, 1.0, 0]])

class_targets = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0]])

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(softmax_outputs, class_targets)
print(loss)
