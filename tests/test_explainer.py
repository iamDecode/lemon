from unittest import TestCase
import numpy as np
import pandas as pd
from scipy.stats import ks_1samp, uniform, truncnorm
from lemon import LemonExplainer
from lemon.kernels import uniform_kernel, gaussian_kernel
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.datasets import load_iris, load_wine, load_linnerud
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from scipy.special import gammainccinv
from functools import partial


# chosen by fair dice roll.
# guaranteed to be random.
random_state = 4


class TestExplainer(TestCase):
  def setUp(self):
    data = load_iris(as_frame=True)
    X = data.data
    y = pd.Series(np.array(data.target_names)[data.target])
    y.name = "Class"

    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X, y)

    self.X = X
    self.y = y
    self.clf = clf

  def test_sample_hypersphere_uniform(self):
    for dimensions in np.linspace(2, 50, 10).astype(int):
      for radius_max in np.linspace(0.25, 1.0, 10):
        X = np.array([
          np.array([-1 for d in range(0, dimensions)]),
          np.array([1 for d in range(0, dimensions)])
        ])
        explainer = LemonExplainer(X, radius_max=radius_max)

        sphere = explainer._sample_hypersphere(10000, dimensions)
        distances = np.linalg.norm(sphere, axis=1) / radius_max

        # Using a uniform kernel, the distance of each point in the sphere to the origin
        # should be uniformly distributed (accounted for sphere divergence).
        test = ks_1samp(distances * distances**(dimensions - 1), uniform.cdf)
        assert test.statistic < 0.05

  def test_sample_hypersphere_size(self):
    for dimensions in np.linspace(2, 50, 10).astype(int):
      for radius_max in np.linspace(0.25, 1.0, 10):
        X = np.array([
          np.array([-1 for d in range(0, dimensions)]),
          np.array([1 for d in range(0, dimensions)])
        ])
        explainer = LemonExplainer(X, radius_max=radius_max)

        sphere = explainer._sample_hypersphere(10000, dimensions)
        distances = np.linalg.norm(sphere, axis=1)

        # All samples should fall strictly within the specified maximum radius
        assert np.isclose(radius_max, np.max(distances), rtol=1e-02)
        assert np.max(distances) < radius_max

  def test_sample_hypersphere_uniform_circle(self):
    for dimensions in np.linspace(2, 50, 10).astype(int):
      for radius_max in np.linspace(0.25, 1.0, 10):
        X = np.array([
          np.array([-1 for d in range(0, dimensions)]),
          np.array([1 for d in range(0, dimensions)])
        ])
        explainer = LemonExplainer(X, radius_max=radius_max)

        sphere = explainer._sample_hypersphere(10000, dimensions)
        axis_aligned_variance = np.var(sphere, axis=0)
        difference = np.abs(
            np.min(axis_aligned_variance) -
            np.max(axis_aligned_variance))

        # All samples should have the same distribution in each direction
        # (rotational symmetry)
        assert difference < 1e-02

  def test_sample_hypersphere_gaussian_circle(self):
    for dimensions in np.linspace(2, 50, 10).astype(int):
      for radius_max in np.linspace(0.25, 1.0, 10):
        X = np.array([
          np.array([-1 for d in range(0, dimensions)]),
          np.array([1 for d in range(0, dimensions)])
        ])
        p = 0.99
        kernel_width = 1 / (2 * gammainccinv(dimensions / 2, (1 - p)))
        kernel = lambda x: gaussian_kernel(x, kernel_width)
        explainer = LemonExplainer(
            X, distance_kernel=kernel, radius_max=radius_max)

        sphere = explainer._sample_hypersphere(10000, dimensions)
        axis_aligned_variance = np.var(sphere, axis=0)
        difference = np.abs(
            np.min(axis_aligned_variance) -
            np.max(axis_aligned_variance))

        # All samples should have the same distribution in each direction
        # (rotational symmetry)
        assert difference < 1e-02

  def test_transform_uniform(self):
    for dimensions in np.linspace(2, 50, 10).astype(int):
      for radius_max in np.linspace(0.25, 1.0, 10):
        X = np.array([
          np.array([-1 for d in range(0, dimensions)]),
          np.array([1 for d in range(0, dimensions)])
        ])
        explainer = LemonExplainer(X, radius_max=radius_max)
        kernel = uniform_kernel
        sampling_kernel = explainer._transform(
            kernel, dimensions, 5000, radius_max, adjust=False)

        samples = sampling_kernel(np.random.uniform(size=50000)) / radius_max

        # Samples should be uniformly distributed
        test = ks_1samp(samples, uniform.cdf)
        assert test.statistic < 0.05

  def test_transform_normal(self):
    for dimensions in np.linspace(2, 50, 10).astype(int):
      for radius_max in np.linspace(0.25, 1.0, 10):
        X = np.array([
          np.array([-1 for d in range(0, dimensions)]),
          np.array([1 for d in range(0, dimensions)])
        ])
        explainer = LemonExplainer(X, radius_max=radius_max)

        p = 0.99
        kernel_width = 1 / (2 * gammainccinv(dimensions / 2, (1 - p)))
        kernel = lambda x: gaussian_kernel(x, kernel_width=kernel_width)

        sampling_kernel = explainer._transform(
            kernel, dimensions, 5000, radius_max, adjust=False)
        samples = sampling_kernel(np.random.uniform(size=50000)) / radius_max

        # Samples should be (truncated)normally distributed
        cdf = partial(truncnorm.cdf, a=0, b=1 / (kernel_width /
                      radius_max), scale=kernel_width / radius_max)
        test = ks_1samp(samples, cdf)
        assert test.statistic < 0.05

  def test_explain_instance(self):
    explainer = LemonExplainer(self.X, random_state=random_state)

    rf_fi = self.clf.feature_importances_

    lemon_fi = np.mean([
      explainer.explain_instance(instance, self.clf.predict_proba)[
          0].feature_contribution
      for _, instance in self.X.iterrows()
    ], axis=0)
    lemon_fi /= np.sum(np.abs(lemon_fi))
    lemon_fi = np.abs(lemon_fi)

    # Mean important features should be roughly proportional to random forest
    # feature importance
    assert (np.array_equal(np.argsort(rf_fi), np.argsort(lemon_fi)))

  def test_explain_instance_scaling_invariant(self):
    # Unscaled
    explainer = LemonExplainer(self.X, random_state=random_state)

    lemon_fi = np.mean([
      explainer.explain_instance(instance, self.clf.predict_proba)[
          0].feature_contribution
      for _, instance in self.X.iterrows()
    ], axis=0)
    lemon_fi /= np.sum(np.abs(lemon_fi))
    lemon_fi = np.abs(lemon_fi)

    # Scaled
    data = load_iris(as_frame=True)
    X_scaled = data.data
    X_scaled.loc[:, 'petal width (cm)'] *= 1000
    y = pd.Series(np.array(data.target_names)[data.target])
    y.name = "Class"

    clf_scaled = RandomForestClassifier(random_state=random_state)
    clf_scaled.fit(X_scaled, y)
    explainer_scaled = LemonExplainer(X_scaled, random_state=random_state)

    lemon_fi_scaled = np.mean([
      explainer_scaled.explain_instance(instance, clf_scaled.predict_proba)[
          0].feature_contribution
      for _, instance in X_scaled.iterrows()
    ], axis=0)
    lemon_fi_scaled /= np.sum(np.abs(lemon_fi_scaled))
    lemon_fi_scaled = np.abs(lemon_fi_scaled)

    # Scaling features should have have no effect on which feature is most
    # important
    assert np.allclose(lemon_fi, lemon_fi_scaled, rtol=0.01, atol=0.01)

  def test_explain_instance_plot(self):
    explainer = LemonExplainer(self.X, random_state=random_state)

    instance = self.X.iloc[-1, :]
    exp = explainer.explain_instance(instance, self.clf.predict_proba)[0]

    exp.show_in_notebook()


class TestExplainerCategorical(TestCase):
  def setUp(self):
    data = load_wine(as_frame=True)
    X = data.data
    X.loc[:, 'ash'] = pd.cut(X.loc[:, 'ash'], 4)
    y = pd.Series(np.array(data.target_names)[data.target])
    y.name = "Class"

    self.categorical_features = [2]
    ordinal_encoder = make_column_transformer(
      (
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan),
        make_column_selector(dtype_include="category"),
      ),
      remainder="passthrough",
      verbose_feature_names_out=False,
    )

    clf = make_pipeline(
      ordinal_encoder, HistGradientBoostingClassifier(
          categorical_features=self.categorical_features, random_state=random_state)
    )
    clf.fit(X, y)

    self.X = X
    self.y = y
    self.clf = clf

  def test_explain_instance_plot(self):
    explainer = LemonExplainer(
        self.X,
        radius_max=0.5,
        categorical_features=self.categorical_features,
        random_state=random_state)

    instance = self.X.iloc[120, :]
    exp = explainer.explain_instance(instance, self.clf.predict_proba)[0]

    exp.show_in_notebook()

  def test_explain_instance_without_training_data(self):
    stats = {
      column: dict(
          zip(
              *
              np.unique(
                  self.X[column],
                  return_counts=True))) if i in self.categorical_features else self.X[column].std(
        ddof=0)
      for i, column in enumerate(self.X.columns)
    }
    explainer = LemonExplainer(
        training_data_stats=stats,
        radius_max=0.5,
        categorical_features=self.categorical_features,
        random_state=random_state)

    instance = self.X.iloc[120, :]
    exp = explainer.explain_instance(instance, self.clf.predict_proba)[0]

    exp.show_in_notebook()


class TestExplainerRegression(TestCase):
  def setUp(self):
    data = load_linnerud(as_frame=True)
    X = data.data
    y = data.target
    y.name = "Target"

    regr = make_pipeline(
        StandardScaler(),
        MultiOutputRegressor(
            LinearRegression()))
    regr.fit(X, y)

    self.X = X
    self.y = y
    self.regr = regr

  def test_explain_instance_plot(self):
    explainer = LemonExplainer(
        self.X,
        radius_max=0.5,
        random_state=random_state)

    instance = self.X.iloc[10, :]
    exp = explainer.explain_instance(instance, self.regr.predict)[0]

    exp.show_in_notebook()

  def test_explain_instance_without_training_data(self):
    stats = {
      column: self.X[column].std(ddof=0)
      for i, column in enumerate(self.X.columns)
    }
    explainer = LemonExplainer(
        training_data_stats=stats,
        radius_max=0.5,
        random_state=random_state)

    instance = self.X.iloc[10, :]
    exp = explainer.explain_instance(instance, self.regr.predict)[0]

    exp.show_in_notebook()
