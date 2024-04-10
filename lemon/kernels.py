import numpy as np


def uniform_kernel(x):
  return 1


def gaussian_kernel(x, kernel_width):
  # https://github.com/marcotcr/lime/blob/fd7eb2e6f760619c29fca0187c07b82157601b32/lime/lime_tabular.py#L251
  return np.sqrt(np.exp(-(x ** 2) / kernel_width ** 2))


def sqcos_kernel(x):
  return np.cos(x) ** 2


def trapezoid_kernel(x, a, b):
  if 0 <= x and x <= a:
    return (2 / (a + b))
  elif a <= x and x <= b:
    return (2 / (a + b)) * ((b - x) / (b - a))
  else:
    return 0
