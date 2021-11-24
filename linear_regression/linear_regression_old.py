import numpy as np
import matplotlib.pyplot as plt

def plot(x,y):
  plt.scatter(x, y, color = "m",
             marker = "o", s = 30)
  plt.show()

def get_gradient_at_b(x, y, b, m):
  N = len(x)
  diff = 0
  for i in range(N):
    x_val = 

def step_gradient(b_current, m_current, x, y, learning_rate):
  b_gradient = get_gradient_at_b(x, y, b_current, m_current)
  m_gradient = get_gradient_at_m(x, y, b_current, m_current)
  b = b_current - (learning_rate * b_gradient)
  m = m_current - (learning_rate * m_gradient)
  return [b, m]

def train(x, y, learning_rate, num_iterations):
  b = 0
  m = 0
  for i in range(num_iterations):
    b,m = step_gradient(b,m,x,y,learning_rate)
  return [b,m]

def main():
  x = np.array([0,1,2,3,4,5,6,7,8,9])
  y = np.array([1,3,2,5,7,8,8,9,10,12])
  b,m = train(x,y)
  # plot(x,y)

main()