#!/usr/bin/python

import random

random.seed(42)
print(random.randint(1, 100))  # Always prints the same number
print(random.randint(1, 100))  # Always prints the same number
print(random.randint(1, 100))  # Always prints the same number
print(random.randint(1, 100))  # Always prints the same number
print(random.randint(1, 100))  # Always prints the same number
print(random.randint(1, 100))  # Always prints the same number

random.seed(42)  # Reset to same seed
print(random.randint(1, 100))  # Prints same number again
print(random.randint(1, 100))  # Prints same number again
print(random.randint(1, 100))  # Prints same number again
print(random.randint(1, 100))  # Prints same number again
print(random.randint(1, 100))  # Prints same number again
print(random.randint(1, 100))  # Prints same number again
