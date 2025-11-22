def fib(n):
    if n <= 0: return 0
    if n == 1: return 1
    return fib(n-1) + fib(n-2)

def is_divisible(n, i):
    return n % i == 0

def is_prime(n):
    # BUG: 1 is NOT a prime number.
    # However, this logic returns True for 1 because the loop range(2, 2) is empty.
    # Correct logic should be: if n <= 1: return False
    if n <= 0: 
        return False
    
    limit = int(n**0.5)
    for i in range(2, limit + 1):
        if is_divisible(n, i):
            return False
    return True

def get_prime_fibs(k):
    found = []
    i = 1
    while len(found) < k:
        val = fib(i)
        if is_prime(val):
            found.append(val)
        i += 1
    return found

def main():
    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21...
    # Primes in sequence: 2, 3, 5, 13... (1 is NOT prime)
    
    # Goal: Find the first 3 prime numbers in the Fibonacci sequence.
    # Expected: [2, 3, 5]
    
    k = 3
    print(f"Searching for the first {k} prime Fibonacci numbers...")
    
    result = get_prime_fibs(k)
    print(f"Result: {result}\n")
    
    # Test case
    expected = [2, 3, 5]
    assert result == expected, f"Expected {expected}, but got {result}"

if __name__ == "__main__":
    main()
