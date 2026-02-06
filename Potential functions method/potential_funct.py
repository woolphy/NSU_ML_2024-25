import numpy as np
from sklearn import metrics


class Potential_method():
    def __init__(self, train_x, train_y, kernel, h, epoch_number) -> None:
        self.classes = np.unique(train_y)
        self.train_x = train_x
        self.train_y = train_y
        self.charges = np.zeros_like(train_y)
        self.indexes = np.arange(0, len(train_y))
        self.Kernel = kernel
        self.h = h
        self.epoch_number = epoch_number

    def distance_computing(self, u, v, p=2):
        return np.sum(((u - v) ** p), -1) ** (1 / p)

    def predict(self, x: np.array):
        test_x = np.copy(x)
        if len(test_x.shape) < 2:
            test_x = test_x[np.newaxis, :]
        u = test_x[:, np.newaxis, :]
        x_u = self.train_x[np.newaxis, :, :]
        weights = self.charges * self.Kernel(self.distance_computing(u, x_u) / self.h)
        table = np.zeros((test_x.shape[0], len(self.classes)))
        for class_ in self.classes:
            table[:, class_] = np.sum(weights[:, self.train_y == class_], axis=1)
        return np.argmax(table, axis=1)

    def fit(self):
        self.charges[0] = 1
        for _ in range(self.epoch_number):
            for i in range(self.train_x.shape[0]):
                if self.predict(self.train_x[i]) != self.train_y[i]:
                    self.charges[i] += 1
        self.train_x = self.train_x[self.charges != 0, ...]
        self.train_y = self.train_y[self.charges != 0, ...]
        self.indexes = self.indexes[self.charges != 0, ...]
        self.charges = self.charges[self.charges != 0, ...]

    def show_accuracy(self, X, y, test_x, test_y):
        predict_arr = self.predict(test_x)
        print("accuracy = ", metrics.accuracy_score(test_y, predict_arr))

    def get_bad_prediction(self, test_x, test_y):
        bad_predictions_array = list()
        predict_arr = self.predict(test_x)
        for i in range(len(test_y)):
            if predict_arr[i] != test_y[i]:
                bad_predictions_array.append(i)
        return bad_predictions_array
