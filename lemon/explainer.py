import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from bisect import bisect
from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler

from .explanation import Explanation
from .kernels import uniform_kernel


class LemonExplainer(object):
  """
  Intantiates the explainer

  Parameters
  ----------
  training_data : numpy.array
      Original training data the reference model (model to be explained) was trained on.

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
      training_data, 
      sample_size=5000, 
      distance_kernel=None, 
      radius_max=1, 
      random_state=None,
  ):
    self.random_state = check_random_state(random_state)
    np.random.seed(random_state)

    self.training_data = training_data
    self.scaler = StandardScaler(with_mean=False)
    self.scaler.fit(training_data)
    
    dimensions = training_data.shape[1]

    if distance_kernel is None:
      distance_kernel = uniform_kernel
  
    self.sampling_kernel = self._transform(distance_kernel, dimensions, radius_max=radius_max)

    # Create hypersphere samples. The sphere is only computed once for performance and stability,
    # alternatively we can resample the sphere every time `explain_instance` is called, but this
    # affects running time. My preliminary tests indicate not resampling every time does not 
    # practically affect the resulting explanation much.
    self.sphere = self._sample_hypersphere(sample_size, dimensions)

  @property
  def surrogate(self):
    """
    Surrogate model. 

    This can be any regression model as long as it exposes 
    feature-wise coefficients in `surrogate.coef_`.
    
    """
    try:
      return self._surrgate
    except AttributeError:
      self._surrogate = Ridge(fit_intercept=True, random_state=self.random_state)
      return self._surrogate

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
    surrogate = surrogate or self.surrogate
  
    # Create transfer dataset by perturbing the original instance with the hypersphere samples
    X_transfer = self.scaler.inverse_transform(self.sphere) + np.array(instance).reshape(1,-1)
    
    
    if isinstance(self.training_data, pd.DataFrame):
      prediction = predict_fn(pd.DataFrame(instance).T)
      y_transfer = predict_fn(pd.DataFrame(X_transfer, columns=self.training_data.columns))
    else:
      prediction = predict_fn(np.array(instance).reshape(1,-1))
      y_transfer = predict_fn(X_transfer)

    if labels is None:
      prediction_index = np.argmax(prediction)
      labels = (prediction_index,)

    def explain_label(label):
      surrogate.fit(X_transfer, y_transfer[:,label])
      certainty = prediction[:,label]
      score = surrogate.score(X_transfer, y_transfer[:,label])

      return Explanation(
        instance,
        surrogate,
        label=label,
        label_certainty=certainty,
        local_faithfulness=score
      )

    return [explain_label(label) for label in labels]

  def _sample_hypersphere(self, sample_size, dimensions):
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

    dimensions: int
        Number of features, equal to the number of features in the original training dataset.

    """
    sphere = np.random.normal(size=(sample_size, dimensions))
    sphere = normalize(sphere)
    sphere *= self.sampling_kernel(np.random.uniform(size=sample_size)).reshape(-1,1)
    
    return sphere
  
  def _transform(self, kernel, dimensions, sample_size=5000, radius_max=1, adjust=True): 
    """
    Inverse transform sampling
    
    Generate samples distributed according to the distribution described by the `kernel` function,
    adjusted for sampling within a hypersphere ($x^{n-1}$).

    Parameters
    ----------
    kernel: callable
        Original PDF to sample from.

    dimensions: int
        Number of dimensions to adjust for hypersphere sampling.

    sample_size: int
        Number of samples to invert the distribution function.

    radius_max: float
        Maximum radius of the hypersphere, in _normalized_ feature space. In other words, how much 
        LEMON varies the feature value of each feature to find similar data points, as a percentage
        of the feature range. The lower, the more local the explanation.

    """
    if adjust:
      pdf = lambda x: kernel(x) * (x ** (dimensions - 1))
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
