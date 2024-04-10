import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams, transforms


class Explanation(object):
  """
  Intantiates the explanation for the provided label

  Parameters
  ----------
  instance: pandas.DataFrame, numpy.array
      The instance that is explained.

  surrogate: sklearn.base.RegressorMixin
      The fitted surrogate model. This can be any regression model as long as it exposes
      feature-wise coefficients in `surrogate.coef_`.

  local_faithfulness: float
      Local faithfulness of the surrogate, or the sklearn score with respect to the
      reference model to be explained.

  label: int
      The class label this explanation explains. Typically the predicted label by the model
      for this instance.

  label_certainty: float
      The predicted probability for the prediction of this instance.

  """

  def __init__(self, instance, surrogate, label,
               label_certainty, local_faithfulness):
    # TODO: check if surrogate is fitted and has coef_
    self.instance = instance
    self.surrogate = surrogate
    self.label = label
    self.label_certainty = label_certainty
    self.local_faithfulness = local_faithfulness

  @property
  def feature_contribution(self):
    """
    Returns a score for each feature that indicates how much it contributed to the final prediction.

    More precisely, it shows the sensitivity of the feature: a small change in an important feature's
    value results in a relatively large change in prediction.

    """
    # TODO: return feature contribution for other types of models.
    return self.surrogate.coef_

  def plot(self, normalize=True, absolute_value=False):
    """
    Plot feature contribution as vertical bar chart.

    """
    if not isinstance(self.instance, pd.Series):
      raise Exception("provide instance as pandas Series to show in notebook.")

    plt.style.use('fivethirtyeight')
    rcParams.update({
      'figure.facecolor': 'white',
      'axes.facecolor': 'white',
      'axes.edgecolor': 'white',
      'savefig.facecolor': 'white'
    })

    feature_contribution = self.feature_contribution

    if absolute_value:
      feature_contribution = np.abs(feature_contribution)
    if normalize:
      feature_contribution /= np.sum(np.abs(feature_contribution))

    contribution = pd.Series(feature_contribution, index=self.instance.index)
    contribution = contribution.iloc[contribution.abs().argsort()]
    fig, _ = plt.subplots()
    contribution.plot.barh(
      color=['#ff5a42' if c < 0 else '#007cff' for c in contribution],
    )
    plt.axvline(x=0, color='black', linewidth=1)

    fig.suptitle("Feature contribution (LEMON)", x=0, y=1,
                 ha='left', va='top', fontweight='bold')
    offsetY = transforms.ScaledTranslation(0, -0.3, fig.dpi_scale_trans)
    plt.title("Shows the sensitivity of a feature: a small change in an important feature's value results in a relatively large change in prediction.",
              x=0, y=1, ha='left', va='top', fontsize=13, color='#5e6e8a', transform=fig.transFigure + offsetY, wrap=True)

    plt.tight_layout()

  def as_list(self):
    """
    Convenience method with similar interface to LIME.

    """
    return self.feature_contribution

  def show_in_notebook(self):
    """
    Convenience method with similar interface to LIME.

    """
    self.plot()
    plt.show()
