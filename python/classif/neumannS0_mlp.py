import torch
import torch.nn as nn
import numpy as np

from ..neumannS0_mlp import Neumann_mlp_base


class Neumann_mlp_clf(Neumann_mlp_base):

    def __init__(self, depth, n_epochs, batch_size, lr, early_stopping, residual_connection, mlp_depth, init_type, verbose):
        super().__init__(depth, n_epochs, batch_size, lr, criterion=nn.BCEWithLogitsLoss(), early_stopping=early_stopping, residual_connection=residual_connection, mlp_depth=mlp_depth, init_type=init_type, verbose=verbose)
        # self.classes = None

        self.bce_train = []
        self.bce_val = []
        # self.acc_train = []
        self.acc_val = []

    # def fit(self, X, y, X_val, y_val):
    #     super().fit(X, y, X_val=X_val, y_val=y_val)
    #     self.classes = np.sort(np.unique(y))
    #     assert len(self.classes) <= 2

    # def predict(self, X):
    #     y_hat = super().predict(X)
    #     pred_classes = self.classes[0]*np.ones_like(pred)
    #     pred_classes[pred >= 0] = self.classes[1]
    #     return pred_classes

    def evaluate_train_loss(self, y_hat, y, running_loss=None):
        bce = self.criterion(y_hat, y).item()
        self.bce_train.append(bce)

        # var = ((y - y.mean())**2).mean()
        # r2 = 1 - mse/var
        # self.r2_train.append(r2)

        # if self.verbose and running_loss is not None:
        #     print("Train loss - r2: {}, mse: {}".format(r2, running_loss/self.batch_size))

    def evaluate_val_loss(self, y_hat, y_val, early_stopping=None):
        with torch.no_grad():
            bce_val = self.criterion(y_hat, y_val).item()
            self.bce_val.append(bce_val)

            acc_val = torch.sum(y_hat == y_val)/y_hat.shape[0]

            if self.verbose:
                print("Validation accuracy is: {}".format(acc_val))

        if self.early_stop:
            early_stopping(bce_val, self.net)
            if early_stopping.early_stop:
                if self.verbose:
                    print("Early stopping")
                return True

        return False