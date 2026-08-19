import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #

        # (1/1+e^-x)
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        

        '''

         (y_true - sigmoid ( dot(x, w) + b ))^2

         w = w * e-x(dot)
        '''

        z = x @ w + b

        e = 1/np.exp(z)

        sigmoid = 1 / (1 + e)

        dL_dw = np.round((-1 * np.square(sigmoid) * e * x) * (y_true - sigmoid), 5)
        dL_db = np.round((-1 * np.square(sigmoid) * e) * (y_true - sigmoid), 5)

        return dL_dw, dL_db
