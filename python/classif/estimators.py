import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

from ..estimators import ConstantImputedLR, EMLR, MICELR


class ConstantImputedLogit(ConstantImputedLR):

    def __init__(self):
        self._reg = LogisticRegression()


class MeanImputedLogit(ConstantImputedLR):

    def __init__(self):
        self._reg = LogisticRegression()
        self._imp = SimpleImputer(strategy='mean', add_indicator=True)

    def fit(self, X, y):
        T = self._imp.fit_transform(X)
        self._reg.fit(T, y)
        return self

    def predict(self, X):
        T = self._imp.transform(X)
        return self._reg.predict(T)


class EMLogit(EMLR):

    def __init__(self):
        super().__init__()
        self.classes = None

    def fit(self, X, y):
        super().fit(X, y)
        self.classes = np.sort(np.unique(y))
        assert len(self.classes) <= 2

    def predict(self, X):
        pred = super().predict(X)
        pred_classes = self.classes[0]*np.ones_like(pred)
        pred_classes[pred >= 0] = self.classes[1]
        return pred_classes


class MICELogit(MICELR):
    
    def __init__(self):
        super().__init__()
        self._reg = LogisticRegression()
    