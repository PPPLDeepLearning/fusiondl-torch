from __future__ import print_function
import global_vars as g
import numpy as np
import abc

from evaluation import mse_np, binary_crossentropy_np, hinge_np


# synchronize output from TensorFlow initialization via Keras backend
if g.comm is not None:
    g.flush_all_inorder()
    g.comm.Barrier()


# Requirement: larger value must mean disruption more likely.
class Target(object):
    activation = "linear"
    loss = "mse"

    @abc.abstractmethod
    def loss_np(y_true, y_pred):
        from conf import conf

        return conf["model"]["loss_scale_factor"] * mse_np(y_true, y_pred)

    @abc.abstractmethod
    def remapper(ttd, T_warning):
        return -ttd

    @abc.abstractmethod
    def threshold_range(T_warning):
        return np.logspace(-1, 4, 100)


class BinaryTarget(Target):
    activation = "sigmoid"
    loss = "binary_crossentropy"

    @staticmethod
    def loss_np(y_true, y_pred):
        from conf import conf

        return conf["model"]["loss_scale_factor"] * binary_crossentropy_np(
            y_true, y_pred
        )

    @staticmethod
    def remapper(ttd, T_warning, as_array_of_shots=True):
        binary_ttd = 0 * ttd
        mask = ttd < np.log10(T_warning)
        binary_ttd[mask] = 1.0
        binary_ttd[~mask] = 0.0
        return binary_ttd

    @staticmethod
    def threshold_range(T_warning):
        return np.logspace(-6, 0, 100)


class TTDTarget(Target):
    activation = "linear"
    loss = "mse"

    @staticmethod
    def loss_np(y_true, y_pred):
        from conf import conf

        return conf["model"]["loss_scale_factor"] * mse_np(y_true, y_pred)

    @staticmethod
    def remapper(ttd, T_warning):
        mask = ttd < np.log10(T_warning)
        ttd[~mask] = np.log10(T_warning)
        return -ttd

    @staticmethod
    def threshold_range(T_warning):
        return np.linspace(-np.log10(T_warning), 6, 100)


class TTDInvTarget(Target):
    activation = "linear"
    loss = "mse"

    @staticmethod
    def loss_np(y_true, y_pred):
        return mse_np(y_true, y_pred)

    @staticmethod
    def remapper(ttd, T_warning):
        eps = 1e-4
        ttd = 10 ** (ttd)
        mask = ttd < T_warning
        ttd[~mask] = T_warning
        ttd = (1.0) / (ttd + eps)  # T_warning
        return ttd

    @staticmethod
    def threshold_range(T_warning):
        return np.logspace(-6, np.log10(T_warning), 100)


class TTDLinearTarget(Target):
    activation = "linear"
    loss = "mse"

    @staticmethod
    def loss_np(y_true, y_pred):
        from conf import conf

        return conf["model"]["loss_scale_factor"] * mse_np(y_true, y_pred)

    @staticmethod
    def remapper(ttd, T_warning):
        ttd = 10 ** (ttd)
        mask = ttd < T_warning
        ttd[~mask] = 0  # T_warning
        ttd[mask] = T_warning - ttd[mask]  # T_warning
        return ttd

    @staticmethod
    def threshold_range(T_warning):
        return np.logspace(-6, np.log10(T_warning), 100)


# implements a "maximum" driven loss function. Only the maximal value in the
# time sequence is punished. Also implements class weighting
# class MaxHingeTarget(Target):
# TODO(KGF): removed because implementation depended on TensorFlow backend


class HingeTarget(Target):
    activation = "linear"

    loss = "hinge"  # hinge

    @staticmethod
    def loss_np(y_true, y_pred):
        from conf import conf

        return conf["model"]["loss_scale_factor"] * hinge_np(y_true, y_pred)
        # return squared_hinge_np(y_true, y_pred)

    @staticmethod
    def remapper(ttd, T_warning, as_array_of_shots=True):
        binary_ttd = 0 * ttd
        mask = ttd < np.log10(T_warning)
        binary_ttd[mask] = 1.0
        binary_ttd[~mask] = -1.0
        return binary_ttd

    @staticmethod
    def threshold_range(T_warning):
        return np.concatenate(
            (
                np.linspace(-2, -1.06, 100),
                np.linspace(-1.06, -0.96, 100),
                np.linspace(-0.96, 2, 50),
            )
        )
