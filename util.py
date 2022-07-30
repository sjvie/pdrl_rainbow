def get_next_power_of_2(k):
    n = 1
    while n < k:
        n *= 2
    return n


def is_power_of_2(k):
    return (k & (k - 1) == 0) and k > 0


class LinearValue:
    def __init__(self, start_value, end_value, step_start, step_end):
        assert step_start < step_end

        self.start_value = start_value
        self.end_value = end_value
        self.step_start = step_start
        self.step_end = step_end

        self.total_steps = self.step_end - self.step_start
        self.diff_value = self.end_value - self.start_value

    def __call__(self, step):
        if step <= self.step_start:
            return self.start_value
        elif step >= self.step_end:
            return self.end_value
        else:
            return self.start_value + self.diff_value * ((step - self.step_start) / self.total_steps)
