import numpy as np

class UniqueRandomGenerator:
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
        self.range_values = list(range(min_value, max_value + 1))
        self.previous_value = None
        self.reset()

    def reset(self):
        """ Shuffle the range values to start fresh """
        np.random.shuffle(self.range_values)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            for value in self.range_values:
                if value != self.previous_value:
                    self.previous_value = value
                    return value
            self.reset()  # Re-shuffle after a full cycle to avoid patterns