import random
import math
from rich.console import Console
from rich.table import Table

console = Console()

table = Table(title="Score to Score")
table.add_column("old knowledge score")
table.add_column("weights")
table.add_column("difficulties")
table.add_column("score in test")
table.add_column("new knowledge score")


def sigmoid(x):
    return 1 / (1 + pow(math.e, -x))


def func1(x):
    return 0.8 * x + 0.2 if x <= 1 else 1


n = 2
F = [0, 0.3, 0.6, 1]
epsilon = 1

# ! include the fact that more concepts = more difficulty
# we have 10 questions
# each question has a certain number of concepts associated with it (fixed)
# each concept has a difficulty score and a weight
# we have 1 student who can get different scores in the test

for i in range(10):
    K = [round(random.random(), 2) for i in range(2)]
    for f in F:
        p = round(random.random(), 2)
        W = [p, round(1 - p, 2)]
        D = [random.randint(1, 5), random.randint(1, 5)]
        K_ = [func1(K[j] + D[j] * W[j] * f / (10 * (K[j] + epsilon)))
              if f != 0 else K[j] for j in range(2)]
        K_ = [round(k_, 2) for k_ in K_]
        table.add_row(str(K), str(W), str(D), str(f), str(K_))
console.print(table)
