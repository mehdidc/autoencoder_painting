import numpy as np

class Invert:
    def __call__(self, x):
        return 1 - x

class Gray:
    def __call__(self, x):
        return x[0:1]
