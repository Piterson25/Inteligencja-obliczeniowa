def prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    for i in range(2, n):
        if n % i == 0:
            return False
    return True


print(prime(3))
print(prime(4))
print(prime(49))


def select_primes(x):
    result = []

    for n in x:
        if prime(n):
            result.append(n)

    return result


print(select_primes([3, 6, 11, 25, 19]))
