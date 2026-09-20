# autodiff

Automatic differentiation (autodiff) works by decomposing a computer program into a sequence of basic arithmetic operations such as +, *, ** and relu, and applying the chain rule in order to calculate the exact derivative.

## Installation

```bash
git clone https://github.com/pablogarciamez/autodiff.git
cd autodiff
pip install -e .
```

## Usage

```python
from autodiff import Value

x = Value(2.0) # Variables
y = Value(-3.0)

f = (x * y).relu() # Sequence of basic arithmetic operations

f.backward() # Autodiff

print(x.grad, y.grad) # Partial derivatives
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```