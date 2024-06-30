import math
import matplotlib.pyplot as plt

eps = 1e-8
forget_rate = 1-10*eps

context_length = 20000000

x = list(range(context_length))
y = [math.pow(forget_rate,i) for i in x]

#plt.figure(figsize=(10,6))
plt.plot(x,y)
plt.show()