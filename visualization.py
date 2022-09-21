import matplotlib.pyplot as plt
import json

colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]

# input parameters
name = 'log02'

# visualization
with open('%s' % name, 'r') as f:
    data = json.load(f)

print('Number of datapoints: %s' % len(data['time']))

plt.plot(data['time'], data['vx'])
plt.plot(data['time'], data['v_ref'])
plt.ylabel('yaw [rad]')
plt.xlabel('time [s]')
plt.show()
