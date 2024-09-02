import numpy as np
x = np.linspace(-2,2,1000)
y1 = np.exp(-x)
y2 = np.exp(x)
import matplotlib.pyplot as plt
plt.plot(x,y1,label= 'gyan')
plt.plot(x,y2,label = 'murkho')
plt.xlabel('As time goes')
plt.ylabel('gyan')

plt.legend()
plt.savefig('brainless_maniac_ok.png')
plt.show()
