from tinygrad.helpers import getenv, colored, trange
from tinygrad.nn.datasets import mnist
from examples.beautiful_mnist import Model
import time

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = mnist()
  
  model = Model()
  res = (model(X_test).argmax(axis=1) == Y_test).mean()*100

  start = time.time()
  list = res.tolist()
  print(f"tolist() took {(time.time() - start):.4f} seconds {list}")