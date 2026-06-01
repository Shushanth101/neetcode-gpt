import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_value = np.max(z)
        shift_z = z - max_value
        sum = np.sum(np.exp(shift_z))
        return np.round(np.exp(shift_z)/sum,4)
        pass
