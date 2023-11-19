import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from bisect import bisect
from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline

from .explanation import Explanation
from .kernels import uniform_kernel


class LemonExplainer(object):
  """
  Intantiates the explainer

  Parameters
  ----------
  training_data : numpy.array
      Original training data the reference model (model to be explained) was trained on.

  categorical_features : list
      list of indices corresponding to the categorical columns. Everything else will be considered
      continuous. Values in these columns MUST be integers.

  training_data_stats: {str: float | dict}
      a dict object having the details of training data statistics. For numerical features, this
      describes the standard deviation, for categorical features the frequencies of each category.

  sample_size: int
      Number of samples to generate to train the surrogate model on. More sample means a 
      more faithful explanation, but longer running time.

  distance_kernel: callable
      Kernel function to determine the density of samples points around the data instance to be 
      explained. Defaults to uniform kernel, which is likely sufficient for most users.

  radius_max: float
      Maximum radius of the hypersphere, in _normalized_ feature space. In other words, how much 
      LEMON varies the feature value of each feature to find 'similar' data points, as a percentage
      of the feature range. The lower, the more local the explanation.

  random_state: int, RandomState instance or None
      Controls the randomness of the hypershphere transfer data samples, and samples used for inverse 
      transform sampling. Can be used for reproducibility of the explanation.

  """
  def __init__(
      self, 
      training_data=None,
      categorical_features=[], 
      training_data_stats={},
      sample_size=5000, 
      distance_kernel=None, 
      radius_max=1, 
      random_state=None,
  ):
    self.random_state = check_random_state(random_state)
    np.random.seed(random_state)

    self.categorical_features = categorical_features

    if training_data is None and len(training_data_stats) > 0:
      self.training_data = pd.DataFrame([], columns=list(training_data_stats.keys()))
      self.scaler = StandardScaler(with_mean=False)
      self.scaler.scale_ = [
        x for i, x in enumerate(training_data_stats.values())
        if i not in categorical_features
      ]
      self.scaler.mean_ = [
        0 for i, _ in enumerate(training_data_stats.values())
        if i not in categorical_features
      ]

      dimensions = len(training_data_stats)
    else:
      self.training_data = training_data

      self.scaler = StandardScaler(with_mean=False)
      self.scaler.fit(np.asanyarray(training_data)[:, np.setdiff1d(np.arange(training_data.shape[1]), categorical_features)])
      
      training_data_stats = {
        training_data.columns[i]: dict(zip(*np.unique(training_data.iloc[:, i], return_counts=True)))
        for i in categorical_features
      }
    
      dimensions = training_data.shape[1]

    n_categorical = len(categorical_features)
    n_numerical = dimensions - n_categorical

    if distance_kernel is None:
      distance_kernel = uniform_kernel
  
    self.sampling_kernel = self._transform(distance_kernel, n_numerical, radius_max=radius_max)
    self.sample_size = sample_size

    # Create hypersphere samples. The sphere is only computed once for performance and stability,
    # alternatively we can resample the sphere every time `explain_instance` is called, but this
    # affects running time. My preliminary tests indicate not resampling every time does not 
    # practically affect the resulting explanation much.
    self.sphere = self._sample_hypersphere(sample_size, n_numerical, categorical_features, training_data_stats)

  @property
  def surrogate(self):
    """
    Surrogate model. 

    This can be any regression model as long as it exposes 
    feature-wise coefficients in `surrogate.coef_`.
    
    """
    try:
      return self._surrogate
    except AttributeError:
      self._surrogate = Ridge(alpha=self.sample_size, fit_intercept=True, random_state=self.random_state)
      return self._surrogate

  @property
  def pipeline(self):
    """
    Pipeline for applying the surrogate model on scaled data.

    Without scaling, the feature contribution is biased towards features with higher values.

    """
    try:
      return self._pipeline
    except AttributeError:
      self._pipeline = make_pipeline(MinMaxScaler(), self.surrogate)
      return self._pipeline

  def explain_instance(self, instance, predict_fn, labels=None, surrogate=None):
    """
    Explain the prediction for the provided instance

    Parameters
    ----------
    instance: numpy.array
        The instance to be explained.

    prediction_fn: callable
        any callable function that returns the predicted probability. Typically the `predict_proba` 
        method from scikit-learn.

    labels: Tuple[int, ...]
        Index of which classes to explain: we calculate the feature contribution towards the prediction 
        of a particular class.

    surrogate: sklearn.base.RegressorMixin
        Use a custom surrogate model. This can be any regression model as long as it exposes 
        feature-wise coefficients in `surrogate.coef_`.
    
    """
    if surrogate is not None:
      self._surrogate = surrogate
  
    # Create transfer dataset by perturbing the original instance with the hypersphere samples
    X_transfer = self._samples_around(instance, self.categorical_features)
    
    if isinstance(self.training_data, pd.DataFrame):
      prediction = predict_fn(pd.DataFrame(instance).T)
      y_transfer = predict_fn(pd.DataFrame(X_transfer, columns=self.training_data.columns))
    else:
      prediction = predict_fn(np.array(instance).reshape(1,-1))
      y_transfer = predict_fn(X_transfer)

    if labels is None:
      prediction_index = np.argmax(prediction)
      labels = (prediction_index,)

    for index in self.categorical_features:
      # single one-hot encoded feature for categorical
      X_transfer[:,index] = (X_transfer[:,index] == instance.values[index]).astype(int)

    def explain_label(label):
      self.pipeline.fit(X_transfer, y_transfer[:,label])
      certainty = prediction[:,label]
      score = self.pipeline.score(X_transfer, y_transfer[:,label])

      return Explanation(
        instance,
        self.surrogate,
        label=label,
        label_certainty=certainty,
        local_faithfulness=score
      )

    return [explain_label(label) for label in labels]
  
  def _samples_around(self, instance, categorical_features):
    instance = np.array(instance).reshape(1,-1)
    
    if categorical_features is not None:
      numeric_idx = [i for i in range(instance.shape[1]) if i not in categorical_features]
      sphere = np.copy(self.sphere)
      sphere[:, numeric_idx] = self.scaler.inverse_transform(sphere[:, numeric_idx])
      sphere[:, numeric_idx] = sphere[:, numeric_idx].astype('float') + instance[:, numeric_idx]
      X_transfer = sphere
    else:
      numeric_idx = [i for i in range(instance.shape[1])]
      X_transfer = self.scaler.inverse_transform(self.sphere) + instance

    return X_transfer

  def _sample_hypersphere(self, sample_size, n_numerical, categorical_features=[], training_data_stats={}):
    """
    Sample from `distance_kernel`-distributed hypersphere.

    Generates samples on a n-dimensional hypersphere distributed according to `distance_kernel` 
    provided in the initializer. This sphere is later scaled and translated into the feature 
    space of the training data set.
    
    Parameters
    ----------
    sample_size: int
        Number of samples to generate to train the surrogate model on. More sample means a 
        more faithful explanation, but longer running time.

    n_numerical: int
        Number of numerical features, equal to the number of numerical features in the original training dataset.

    categorical_features : list
        list of indices corresponding to the categorical columns. Everything else will be considered
        continuous. Values in these columns MUST be integers.

    training_data_stats: {str: float | dict}
        a dict object having the details of training data statistics. For this method, only info
        about the categorical features is used: the frequencies of each category.

    """
    sphere = np.random.normal(size=(sample_size, n_numerical))
    sphere = normalize(sphere)
    sphere *= self.sampling_kernel(np.random.uniform(size=sample_size)).reshape(-1,1)

    for index in categorical_features:
      descriptor = training_data_stats[self.training_data.columns[index]]
      categories = list(descriptor.keys())
      frequencies = np.array(list(descriptor.values()))
      values = self.random_state.choice(categories, size=sample_size, replace=True, p=frequencies / float(sum(frequencies)))
      sphere = np.c_[sphere[:,0:index], values, sphere[:, index:]]

    return sphere
  
  def _transform(self, kernel, n_numerical, sample_size=5000, radius_max=1, adjust=True): 
    """
    Inverse transform sampling
    
    Generate samples distributed according to the distribution described by the `kernel` function,
    adjusted for sampling within a hypersphere ($x^{n-1}$).

    Parameters
    ----------
    kernel: callable
        Original PDF to sample from.

    n_numerical: int
        Number of (numerical) dimensions to adjust for hypersphere sampling.

    sample_size: int
        Number of samples to invert the distribution function.

    radius_max: float
        Maximum radius of the hypersphere, in _normalized_ feature space. In other words, how much 
        LEMON varies the feature value of each feature to find similar data points, as a percentage
        of the feature range. The lower, the more local the explanation.

    """
    if adjust:
      pdf = lambda x: kernel(x) * (x ** (n_numerical - 1))
    else:
      pdf = lambda x: float(kernel(x))

    cdf_samples = np.array([
      pdf(x)
      for x in np.linspace(0, radius_max, sample_size)
    ])

    # Normalize
    cdf_samples = np.cumsum(cdf_samples)
    cdf_samples /= cdf_samples[-1]

    # Invert
    function = lambda y: radius_max * (bisect(cdf_samples, y) / sample_size)

    return np.vectorize(function)
