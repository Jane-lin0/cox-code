from scipy.optimize import minimize


def coordinate_descent(f, initial_guess, tol=1e-6, max_iter=1000):
    x, y = initial_guess
    iter_count = 0
    while iter_count < max_iter:
        # Fix x, optimize y
        y = minimize(lambda y: f(x, y), y).x[0]

        # Fix y, optimize x
        x = minimize(lambda x: f(x, y), x).x[0]

        # Calculate the change in function value
        f_val = f(x, y)

        # Check for convergence
        if abs(f_val - f(x, y)) < tol:
            break

        iter_count += 1

    return (x, y), f(x, y)


# Example function
def example_function(x, y):
    return (x - 2) ** 2 + (y + 3) ** 2


# Initial guess
initial_guess = (0, 0)

# Run coordinate descent algorithm
result, min_value = coordinate_descent(example_function, initial_guess)

print("Minimum found at:", result)
print("Minimum value:", min_value)