def get_next_power_of_2(k):
    n = 1
    while n < k:
        n *= 2
    return n


def is_power_of_2(k):
    return (k & (k - 1) == 0) and k > 0
