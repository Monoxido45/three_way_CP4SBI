from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
from sklearn.base import clone
import torch
from copy import deepcopy
from scipy.stats import gaussian_kde


# defining score basic class
class sbi_Scores(ABC):
    """
    Base class to build any conformity score of choosing.
    In this class, one can define any conformity score for any base model of interest, already fitted or not.
    ----------------------------------------------------------------
    """

    def __init__(self, inference_obj, is_fitted=False, cuda=False, density_obj=None):
        self.inference_obj = inference_obj
        self.is_fitted = is_fitted
        self.cuda = cuda
        self.density = density_obj

    @abstractmethod
    def fit(self, X, theta):
        """
        Fit the base model to training data
        --------------------------------------------------------
        Input: (i)    X: Training feature matrix.
               (ii)   theta: Training parameter vector.

        Output: Scores object
        """
        pass

    @abstractmethod
    def compute(self, X_calib, theta_calib, one_X=False):
        """
        Compute the conformity score in the calibration set
        --------------------------------------------------------
        Input: (i)    X_calib: Calibration feature matrix
               (ii)   theta_calib: Calibration label vector

        Output: Conformity score vector
        """
        pass


class KDE_HPDScore(sbi_Scores):
    def fit(self, X=None, thetas=None, **kwargs):
        # setting up model for SBI package
        if not self.is_fitted:
            if not isinstance(X, torch.Tensor) or X.dtype != torch.float32:
                X = torch.tensor(X, dtype=torch.float32)
            if not isinstance(thetas, torch.Tensor) or thetas.dtype != torch.float32:
                thetas = torch.tensor(thetas, dtype=torch.float32)
            self.inference_obj.append_simulations(thetas, X)
            density = self.inference_obj.train()
            self.posterior = self.inference_obj.build_posterior(density, **kwargs)
        else:
            if self.density is None:
                self.posterior = self.inference_obj.build_posterior(**kwargs)
            else:
                density = deepcopy(self.density)
                self.posterior = self.inference_obj.build_posterior(density, **kwargs)
        return self

    def compute(self, X_calib, thetas_calib, one_X=False, B=1000):
        if not isinstance(X_calib, torch.Tensor) or X_calib.dtype != torch.float32:
            X_calib = torch.tensor(X_calib, dtype=torch.float32)
        if (
            not isinstance(thetas_calib, torch.Tensor)
            or thetas_calib.dtype != torch.float32
        ):
            thetas_calib = torch.tensor(thetas_calib, dtype=torch.float32)

        if self.cuda:
            X_calib = X_calib.to(device="cuda")
            thetas_calib = thetas_calib.to(device="cuda")

        # obtaining posterior estimators
        if not one_X:
            par_n = thetas_calib.shape[0]
            prob_array = np.zeros(par_n)
            for i in tqdm(range(par_n), desc="Computing KDE HPD scores"):
                # fitting KDE for each X
                # sampling
                sample_generated = (
                    self.posterior.sample(
                        (B,),
                        x=X_calib[i].reshape(1, -1),
                        show_progress_bars=False,
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )

                # fitting KDE
                kde = gaussian_kde(sample_generated.T, bw_method="scott")

                # computing log_prob for theta_calib
                prob_array[i] = kde(thetas_calib[i].reshape(1, -1).numpy().T)[0]

        else:
            if len(X_calib.shape) == 1:
                X_calib = X_calib.reshape(1, -1)

            sample_generated = (
                self.posterior.sample(
                    (B,),
                    x=X_calib,
                    show_progress_bars=False,
                )
                .cpu()
                .detach()
                .numpy()
            )

            # fitting KDE
            kde = gaussian_kde(sample_generated.T, bw_method="scott")

            # computing log_prob for only one X
            prob_array = kde(thetas_calib.numpy().T)

        # computing posterior density for theta
        return -(prob_array)


# HPD score
class HPDScore(sbi_Scores):
    def fit(self, X=None, theta=None, **kwargs):
        # setting up model for SBI package
        if not self.is_fitted:
            if not isinstance(X, torch.Tensor) or X.dtype != torch.float32:
                X = torch.tensor(X, dtype=torch.float32)
            if not isinstance(theta, torch.Tensor) or theta.dtype != torch.float32:
                theta = torch.tensor(theta, dtype=torch.float32)
            self.inference_obj.append_simulations(theta, X)
            density = self.inference_obj.train()
            self.posterior = self.inference_obj.build_posterior(density, **kwargs)
        else:
            if self.density is None:
                self.posterior = self.inference_obj.build_posterior(**kwargs)
            else:
                print("Using pre-trained density estimator")
                density = deepcopy(self.density)
                self.posterior = self.inference_obj.build_posterior(density, **kwargs)
        return self

    def compute(self, X_calib, thetas_calib, one_X=False):
        if not isinstance(X_calib, torch.Tensor) or X_calib.dtype != torch.float32:
            X_calib = torch.tensor(X_calib, dtype=torch.float32)
        if (
            not isinstance(thetas_calib, torch.Tensor)
            or thetas_calib.dtype != torch.float32
        ):
            thetas_calib = torch.tensor(thetas_calib, dtype=torch.float32)

        if self.cuda:
            X_calib = X_calib.to(device="cuda")
            thetas_calib = thetas_calib.to(device="cuda")

        # obtaining posterior estimators
        if not one_X:
            par_n = thetas_calib.shape[0]
            log_prob_array = np.zeros(par_n)
            for i in range(par_n):
                log_prob_array[i] = (
                    self.posterior.log_prob(
                        thetas_calib[i].reshape(1, -1),
                        x=X_calib[i].reshape(1, -1),
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )

        else:
            if len(X_calib.shape) == 1:
                X_calib = X_calib.reshape(1, -1)

            # computing log_prob for only one X
            log_prob_array = (
                self.posterior.log_prob(
                    thetas_calib,
                    x=X_calib,
                )
                .cpu()
                .detach()
                .numpy()
            )
        # computing posterior density for theta
        return -(np.exp(log_prob_array))


# Waldo score
class WALDOScore(sbi_Scores):
    def fit(self, X=None, theta=None, **kwargs):
        # setting up model for SBI package
        if not self.is_fitted:
            if not isinstance(X, torch.Tensor) or X.dtype != torch.float32:
                X = torch.tensor(X, dtype=torch.float32)
            if not isinstance(theta, torch.Tensor) or theta.dtype != torch.float32:
                theta = torch.tensor(theta, dtype=torch.float32)
            if self.density is None:
                self.inference_obj.append_simulations(theta, X)
                density = self.inference_obj.train()
            else:
                density = deepcopy(self.density)

        self.posterior = self.inference_obj.build_posterior(density, **kwargs)
        return self

    def compute(self, X_calib, thetas_calib, one_X=False, B=1000, trace=True):
        if not isinstance(X_calib, torch.Tensor) or X_calib.dtype != torch.float32:
            X_calib = torch.tensor(X_calib, dtype=torch.float32)
        if (
            not isinstance(thetas_calib, torch.Tensor)
            or thetas_calib.dtype != torch.float32
        ):
            thetas_calib = torch.tensor(thetas_calib, dtype=torch.float32)

        if self.cuda:
            X_calib = X_calib.to(device="cuda")
            thetas_calib = thetas_calib.to(device="cuda")

        # function to compute waldo
        def compute_waldo(B, X_calib, theta_calib):
            sample_generated = (
                self.posterior.sample(
                    (B,),
                    x=X_calib,
                    show_progress_bars=False,
                )
                .cpu()
                .detach()
                .numpy()
            )

            # computing mean and covariance matrix
            mean_array = np.mean(sample_generated, axis=0)
            covariance_matrix = np.cov(sample_generated, rowvar=False)

            if mean_array.shape[0] > 1:
                waldo_value = (
                    (mean_array - theta_calib).transpose()
                    @ np.linalg.inv(covariance_matrix)
                    @ (mean_array - theta_calib)
                )
            else:
                waldo_value = (mean_array - thetas_calib) ** 2 / (covariance_matrix)
            return waldo_value

        # simulating samples for each X
        if not one_X:
            par_n = thetas_calib.shape[0]
            waldo_array = np.zeros(par_n)

            if trace:
                for i in tqdm(range(par_n), desc="Computing WALDO scores"):
                    if thetas_calib.shape[1] == 1:
                        theta_fixed = thetas_calib[i].cpu().numpy()
                    else:
                        theta_fixed = thetas_calib[i, :].cpu().numpy()

                    waldo_array[i] = compute_waldo(
                        B,
                        X_calib[i, :].reshape(1, -1),
                        theta_fixed,
                    )
            else:
                for i in range(par_n):
                    if thetas_calib.shape[1] == 1:
                        theta_fixed = thetas_calib[i].cpu().numpy()
                    else:
                        theta_fixed = thetas_calib[i, :].cpu().numpy()

                    waldo_array[i] = compute_waldo(
                        B,
                        X_calib[i, :].reshape(1, -1),
                        theta_fixed,
                    )

        else:
            par_n = thetas_calib.shape[0]
            waldo_array = np.zeros(par_n)

            # computing log_prob for only one X
            sample_generated = (
                self.posterior.sample(
                    (B,),
                    x=X_calib.reshape(1, -1),
                    show_progress_bars=False,
                )
                .cpu()
                .detach()
                .numpy()
            )

            # computing mean and covariance matrix
            mean_array = np.mean(sample_generated, axis=0)
            covariance_matrix = np.cov(sample_generated, rowvar=False)
            inv_matrix = np.linalg.inv(covariance_matrix)

            for i in range(par_n):
                if mean_array.shape[0] > 1:
                    theta_fixed = thetas_calib[i, :].cpu().numpy()
                    waldo_array[i] = (
                        (mean_array - theta_fixed).transpose()
                        @ inv_matrix
                        @ (mean_array - theta_fixed)
                    )
                else:
                    theta_fixed = thetas_calib[i].cpu().numpy()
                    waldo_array[i] = (mean_array - theta_fixed) ** 2 / (
                        covariance_matrix
                    )

        # computing posterior density for theta
        return waldo_array


# QuantileScore
class QuantileScore(sbi_Scores):
    def fit(self, X=None, thetas=None):
        # setting up model for SBI package
        if not self.is_fitted:
            if not isinstance(X, torch.Tensor) or X.dtype != torch.float32:
                X = torch.tensor(X, dtype=torch.float32)
            if not isinstance(thetas, torch.Tensor) or thetas.dtype != torch.float32:
                thetas = torch.tensor(thetas, dtype=torch.float32)
            self.inference_obj.append_simulations(thetas, X)
            self.inference_obj.train()

        self.posterior = self.inference_obj.build_posterior()
        return self

    def compute(self, X_calib, thetas_calib, prob=[0.025, 0.095], n_sims=10000):

        if not isinstance(X_calib, torch.Tensor) or X_calib.dtype != torch.float32:
            X_calib = torch.tensor(X_calib, dtype=torch.float32)
        if (
            not isinstance(thetas_calib, torch.Tensor)
            or thetas_calib.dtype != torch.float32
        ):
            thetas_calib = torch.tensor(thetas_calib, dtype=torch.float32)

        # computing quantiles for prob
        par_n = thetas_calib.shape[0]
        quantile_array = np.zeros(par_n)
        for i in range(par_n):
            quantiles_samples_theta = np.quantile(
                self.posterior.sample((n_sims,), x=X_calib[i, :]), q=prob, axis=0
            )

            quantile_array[i] = (
                np.max(
                    quantiles_samples_theta[:, 0] - thetas_calib[i, :],
                    thetas_calib[i, :] - quantiles_samples_theta[:, 1],
                )
                .detach()
                .numpy()
            )

        # computing quantile score posterior for theta
        return quantile_array
