<img width="250" src="https://explaining.ml/images/lemon-logo.png" />

[![PyPI version](https://badge.fury.io/py/lemon_explainer.svg)](https://badge.fury.io/py/lemon_explainer)

**LEMON** is a technique to explain why predictions of machine learning models are made. It does so by providing feature contribution: a score for each feature that indicates how much it contributed to the final prediction. More precisely, it shows the sensitivity of the feature: a small change in an important feature's value results in a relatively large change in prediction. It is similar to the popular [LIME](https://github.com/marcotcr/lime) explanation technique, but is more faithful to the reference model, especially for larger datasets. 

[Website ↗](https://explaining.ml/lemon)
[Academic paper ↗](https://link.springer.com/chapter/10.1007/978-3-031-30047-9_7)


## Installation

To install use pip:

```
$ pip install lemon-explainer
```

## Example
A minimal working example is shown below:

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from lemon import LemonExplainer

# Load dataset
data = load_iris(as_frame=True)
X = data.data
y = pd.Series(np.array(data.target_names)[data.target])

# Train complex model
clf = RandomForestClassifier()
clf.fit(X, y)

# Explain instance
explainer = LemonExplainer(X, radius_max=0.5)
instance = X.iloc[-1, :]
explanation = explainer.explain_instance(instance, clf.predict_proba)[0]
explanation.show_in_notebook()
```

## Development

For a development installation (requires npm or yarn),

```
$ git clone https://github.com/iamDecode/lemon.git
$ cd lemon
```

You may want to (create and) activate a virtual environment:

```
$ python3 -m venv venv
$ source venv/bin/activate
```

Install requirements:

```
$ pip install -r requirements.txt
```

And run the tests with:

```
$ pytest .
```

## Approximate distance kernel LIME

If you prefer to use a Gaussian distance kernel as used in LIME, we can approximate this behavior with:

```python
from lemon import LemonExplainer, gaussian_kernel
from scipy.special import gammainccinv

DIMENSIONS = X.shape[1]
KERNEL_SIZE = np.sqrt(DIMENSIONS) * .75  # kernel size as used in LIME

# Obtain a distance kernel very close to LIME's gaussian kernel, see the paper for details.
p = 0.999
radius = KERNEL_SIZE * np.sqrt(2 * gammainccinv(DIMENSIONS / 2, (1 - p)))
kernel = lambda x: gaussian_kernel(x, KERNEL_SIZE)

explainer = LemonExplainer(X, distance_kernel=kernel, radius_max=radius)
```

This behavior is as close as possible to LIME, but still yields more faithful explanations due to LEMON's improved sampling technique. Read the paper for more details about this approach.

## Citation

If you want to refer to our explanation technique, please cite our paper using the following BibTeX entry:

```bibtex
@inproceedings{collaris2023lemon,
  title={{LEMON}: Alternative Sampling for More Faithful Explanation Through Local Surrogate Models},
  author={Collaris, Dennis and Gajane, Pratik and Jorritsma, Joost and van Wijk, Jarke J and Pechenizkiy, Mykola},
  booktitle={Advances in Intelligent Data Analysis XXI: 21st International Symposium on Intelligent Data Analysis (IDA 2023)},
  pages={77--90},
  year={2023},
  organization={Springer}
}
```

## License

This project is licensed under the BSD 2-Clause License - see the [LICENSE](LICENSE) file for details.
