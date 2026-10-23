

import numpy as np
import matplotlib.pyplot as plt


class MathFunction:
    

    def __init__(self, name, function, color, linestyle="-"):
    
        self.name = name
        self.function = function
        self.color = color
        self.linestyle = linestyle

    def evaluate(self, x):
        
        return self.function(x)

    def derivative(self, x, h=1e-5):
       
        return (self.evaluate(x + h) - self.evaluate(x - h)) / (2 * h)

    def plot(self, ax, x_range):
    
        x_values = np.linspace(x_range[0], x_range[1], 400)
        y_values = self.evaluate(x_values)
        ax.plot(
            x_values,
            y_values,
            label=self.name,
            color=self.color,
            linestyle=self.linestyle,
        )


# Step 1: Define the mathematical functions.
# Use NumPy's sin and exp so each function works with arrays of x values.
def quadratic(x):
    return x**2


def sine(x):
    return np.sin(x)


def exponential(x):
    return np.exp(x)


# Step 2: Create one MathFunction object for each function.
functions = [
    MathFunction("Quadratic: x²", quadratic, color="red", linestyle="-"),
    MathFunction("Sine: sin(x)", sine, color="green", linestyle="--"),
    MathFunction("Exponential: eˣ", exponential, color="blue", linestyle="-."),
]


# Step 3: Choose the interval required by the assignment.
x_range = (-2, 2)


# Step 4: Create two side-by-side plotting areas.
fig, (ax_functions, ax_derivatives) = plt.subplots(1, 2, figsize=(14, 5))

# Left subplot: plot the original functions.
for math_function in functions:
    math_function.plot(ax_functions, x_range)

ax_functions.set_title("Functions Comparison")
ax_functions.set_xlabel("x")
ax_functions.set_ylabel("f(x)")
ax_functions.grid(True)
ax_functions.legend()


x_values = np.linspace(x_range[0], x_range[1], 400)
for math_function in functions:
    ax_derivatives.plot(
        x_values,
        math_function.derivative(x_values),
        label=f"{math_function.name} derivative",
        color=math_function.color,
        linestyle=math_function.linestyle,
    )

ax_derivatives.set_title("Derivatives Comparison (Numerical)")
ax_derivatives.set_xlabel("x")
ax_derivatives.set_ylabel("f'(x)")
ax_derivatives.grid(True)
ax_derivatives.legend()

fig.tight_layout()

# Save a copy of the figure in the same folder as this script.
plt.savefig("math_function_comparison.png", dpi=150, bbox_inches="tight")
plt.show()


# Step 5: Optional quick test of evaluate().
test_x = np.array([-2, -1, 0, 1, 2])
print("Test x values:", test_x)
for math_function in functions:
    print(math_function.name, "=>", math_function.evaluate(test_x))
