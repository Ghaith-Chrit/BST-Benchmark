import math
import string
import random
import numpy as np
from typing import List, Optional, Set, Tuple


def _random_string(length: int = 12) -> str:
    """
    Generate a pseudo-random string-like string containing letters and digits.

    Args:
        length (int, optional): Length of the generated string. Defaults to 12.

    Returns:
        str: Randomly generated string composed of ASCII letters and digits.
    """
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choices(alphabet, k=length))


def generate_strings(
    num: int, length: int = 10, seed: Optional[int] = None
) -> Tuple[List[str], int]:
    """
    Generate a list of unique pseudo-random strings, automatically increasing
    the length if needed to reduce the probability of collisions (birthday paradox).

    Args:
        num (int): Number of unique strings to generate.
        length (int, optional): Initial desired length of each string. Defaults to 10.
        seed (int | None, optional): Seed for the random number generator to ensure reproducibility. Defaults to None.

    Returns:
        Tuple[List[str], int]:
            - List of unique strings composed of letters and digits.
            - The actual length used for strings (may be increased from `length` to avoid collisions).
    """
    if seed is not None:
        random.seed(seed)

    alphabet_size = 62
    target_collision_prob = 1e-6

    # Compute minimal length to satisfy birthday problem
    min_length = math.ceil(
        math.log2(np.int64(num) ** 2 / (2 * target_collision_prob))
        / math.log2(alphabet_size)
    )

    if length < min_length:
        print(
            f"\tString length {length:,} too short for {num:,} strings. "
            f"Increasing to {min_length:,} to reduce collision probability."
        )
        length = min_length

    strings: Set[str] = set()
    while len(strings) < num:
        strings.add(_random_string(length))
    strings_list = list(strings)
    random.shuffle(strings_list)
    return strings_list, length
