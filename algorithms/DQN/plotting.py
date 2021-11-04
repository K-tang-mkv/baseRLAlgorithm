import matplotlib.pyplot as plt

def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    plt.title("Playing")
    plt.xlabel("episodes")
    plt.ylabel("durations")

    plt.plot(episode_durations)
    plt.pause(0.01)
