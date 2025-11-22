def is_prime(n):
  """Checks if a number is prime."""
  if n <= 1:
    return False
  for i in range(2, int(n**0.5) + 1):
    if n % i == 0:
      return False
  return True

def find_primes_between(start, end):
  """Finds all prime numbers between two numbers."""
  primes = []
  for num in range(start, end + 1):
    if is_prime(num):
      primes.append(num)
  return primes

# Example usage:
start_num = 10
end_num = 50
prime_numbers = find_primes_between(start_num, end_num)

# Tests
assert find_primes_between(10, 50) == [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
assert find_primes_between(1, 10) == [2, 3, 5, 7]
assert find_primes_between(20, 30) == [23, 29]

