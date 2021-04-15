"""
An example showing how to run a simulation.
"""

import CellTransmissionModel as ctm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load the network and scenario configurations
net = ctm.Network.from_yaml("test_net.yaml")
sim = ctm.Simulation(net, start_time=0, end_time=24, step_size=1/30)
sim.load_scenario_from_file("test_scenario.yaml")


# Define an animation function to use with matplotlib's FuncAnimation
def anim(t, ax, sim):
    artists = sim.plot(ax, exaggeration=1000)
    sim.step()
    return artists


# Set up and run the animation
fig, ax = plt.subplots()
net.plot_colorbar(ax)
a = FuncAnimation(fig, anim, fargs=(ax, sim), frames=sim.time_steps, repeat=False, blit=True, interval=100)
plt.show()
