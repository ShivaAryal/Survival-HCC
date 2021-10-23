import warnings
import numpy as np
import pandas as pd
from functools import partial
from keras import backend as K
from scipy import spatial, cluster
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from .autoEncoderArchitectures import deep_vae, train_model
from .utils import select_clinical_factors, compute_harrells_c

class MainModel(BaseEstimator):
    # trains a vae to find latent factors

    def __init__(
        self,
        n_hidden=None,
        n_latent=80,
        batch_size=100,
        epochs=400,
        architecture="deep",
        initial_beta_val=0,
        kappa=1.0,
        max_beta_val=1,
        learning_rate=0.0005,
        epsilon_std=1.0,
        batch_normalize_inputs=True,
        batch_normalize_intermediaries=True,
        batch_normalize_embedding=True,
        relu_intermediaries=True,
        relu_embedding=True,
        input_dim=None,
        verbose=0
    ):
        if n_hidden is None:
            n_hidden = [1500]
        self.init_args = {k: v for k, v in locals().items() if k != "self"}
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.batch_size = batch_size
        self.epochs = epochs

        self.architecture = partial(
            deep_vae,
            initial_beta_val=initial_beta_val,
            kappa=kappa,
            max_beta_val=max_beta_val,
            learning_rate=learning_rate,
            epsilon_std=epsilon_std,
            batch_normalize_inputs=batch_normalize_inputs,
            relu_embedding=relu_embedding,
        )

        if input_dim is not None:
            vae, encoder, sampling_encoder, decoder, beta = self.architecture(
                input_dim,
                hidden_dims=self.n_hidden,
                latent_dim=self.n_latent,
                batch_size=self.batch_size,
                epochs=self.epochs,
            )
            self.beta = beta
            self.vae = vae
            self.encoder = encoder
            self.sampling_encoder = sampling_encoder
            self.decoder = decoder

        self.training_fn = partial(
            train_model,
            epochs=epochs,
            batch_size=batch_size,
            kappa=kappa,
            max_beta_val=max_beta_val,
            verbose=verbose,
        )

        self.feature_names = None


    def fit(self, X, y=None, X_validation=None, *args, **kwargs):
        # train autoencoder model

        x_ = self._dict2array(X)
        self._validate_indices(x_)
        self.x_ = x_

        self.nwp_ = self.feature_correlations_ = self.w_ = None

        x_test = self._dict2array(X_validation) if X_validation else self.x_

        if self.feature_names != x_test.columns.tolist():
            raise ValueError("Feature mismatch between X and X_val!")

        if not hasattr(self, "vae"):
            vae, encoder, sampling_encoder, decoder, beta = self.architecture(
                self.x_.shape[1],
                hidden_dims=self.n_hidden,
                latent_dim=self.n_latent,
                batch_size=self.batch_size,
                epochs=self.epochs,
            )
            self.init_args["input_dim"] = self.x_.shape[1]
            self.beta = beta
            self.vae = vae
            self.encoder = encoder
            self.sampling_encoder = sampling_encoder
            self.decoder = decoder
        hist = self.training_fn(
            vae=self.vae, x_train=self.x_, x_val=x_test, beta=self.beta, **kwargs
        )
        self.hist = pd.DataFrame(hist.history)
        return self

    def transform(self, X, encoder="mean"):
        if encoder == "mean":
            the_encoder = self.encoder
        elif encoder == "sample":
            the_encoder = self.sampling_encoder
        else:
            raise ValueError("`encoder` must be one of 'mean' or 'sample'")

        x_ = self._dict2array(X)
        self._validate_indices(x_)
        self.x_ = x_
        self.z_ = pd.DataFrame(
            the_encoder.predict(self.x_),
            index=self.x_.index,
            columns=[f"LF{i}" for i in range(1, self.n_latent + 1)],
        )

        self.feature_correlations_ = None
        self.w_ = None
        return self.z_

    def fit_transform(self, X, y=None, X_validation=None, encoder="mean"):
        self.fit(X, X_validation=X_validation, y=y)
        return self.transform(X, encoder=encoder)

    def cluster(
        self,
        k=None,
        optimal_k_method="ami",
        optimal_k_range=range(3, 10),
        ami_y=None,
        kmeans_kwargs=None,
    ):
        if kmeans_kwargs is None:
            kmeans_kwargs = {"n_init": 1000, "n_jobs": 2}
        if k is not None:
            return pd.Series(
                KMeans(k, **kmeans_kwargs).fit_predict(self.z_), index=self.z_.index
            )
        else:
            from sklearn.metrics import adjusted_mutual_info_score

            if ami_y is None:
                raise Exception(
                    "Must provide ``ami_y`` if using 'ami' to select optimal K."
                )
            z_to_use = self.z_.loc[ami_y.index]
            scorer = lambda yhat: adjusted_mutual_info_score(ami_y, yhat)
            yhats = {
                k: pd.Series(
                    KMeans(k, **kmeans_kwargs).fit_predict(z_to_use),
                    index=z_to_use.index,
                )
                for k in optimal_k_range
            }
            score_name = (
                optimal_k_method
                if isinstance(optimal_k_method, str)
                else optimal_k_method.__name__
            )
            self.kmeans_scores = pd.Series(
                [scorer(yhats[k]) for k in optimal_k_range],
                index=optimal_k_range,
                name=score_name,
            )
            self.kmeans_scores.index.name = "K"
            opt_k_index = np.argmax(self.kmeans_scores)
            self.optimal_k_ = self.kmeans_scores.index[opt_k_index]
            self.yhat_ = yhats[opt_k_index]
            return self.yhat_

    def select_clinical_factors(
        self,
        survival,
        duration_column="duration",
        observed_column="observed",
        alpha=0.05,
        cox_penalizer=0,
    ): 
        # select latent factors which are predictivive of survival

        self.z_clinical_ = select_clinical_factors(
            self.z_,
            survival,
            duration_column=duration_column,
            observed_column=observed_column,
            alpha=alpha,
            cox_penalizer=cox_penalizer,
        )
        return self.z_clinical_

    def c_index(
        self,
        survival,
        clinical_only=True,
        duration_column="duration",
        observed_column="observed",
        cox_penalties=None,
        cv_folds=5,
        sel_clin_alpha=0.05,
        sel_clin_penalty=0,
    ):
        #Compute's Harrell's c-Index
        if cox_penalties is None:
            cox_penalties = [0.1, 1, 10, 100, 1000, 10000]
        if clinical_only:
            z = self.select_clinical_factors(
                survival,
                duration_column,
                observed_column,
                sel_clin_alpha,
                sel_clin_penalty,
            )
        else:
            z = self.z_
        return compute_harrells_c(
            z, survival, duration_column, observed_column, cox_penalties, cv_folds
        )

    def _dict2array(self, X):
        self._validate_X(X)
        new_feature_names = [f"{k}: {c}" for k in sorted(X.keys()) for c in X[k].index]
        sample_names = X[list(X.keys())[0]].columns
        ret = pd.DataFrame(
            np.vstack([X[k] for k in sorted(X.keys())]).T,
            index=sample_names,
            columns=new_feature_names,
        )
        if self.feature_names is not None:
            if set(self.feature_names) != set(ret.columns):
                raise ValueError("Feature mismatch!")
            ret = ret.loc[:, self.feature_names]
        return ret

    def _validate_X(self, X):
        if not isinstance(X, dict):
            raise ValueError("data must be a dict")

        df1 = X[list(X.keys())[0]]
        if any(df.columns.tolist() != df1.columns.tolist() for df in X.values()):
            raise ValueError("All dataframes must have same samples (columns)")

        if any(len(df.index) == 0 for df in X.values()):
            raise ValueError("One of the DataFrames was empty.")

        return True

    def _validate_indices(self, x):
        if self.feature_names is None:
            self.feature_names = x.columns.tolist()
        else:
            if self.feature_names != x.columns.tolist():
                raise ValueError("Feature mismatch with previously fit data!")