import numpy as np

class EnsembleModel:
    def __init__(self, models, weights, class_labels):
        self.models = models
        self.weights = weights
        self.class_labels = class_labels
        self.total_weight = sum(weights)

    def predict(self, img_input):
        final_prob = np.zeros_like(self.models[0].predict(img_input))
        for model, weight in zip(self.models, self.weights):
            final_prob += model.predict(img_input) * weight
        final_prob /= self.total_weight

        idx = np.argmax(final_prob, axis=1)[0]
        return self.class_labels[idx], final_prob
