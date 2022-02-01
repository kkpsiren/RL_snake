import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
from IPython.utils import io

plt.ion()


def plot(scores, mean_scores):
    with io.capture_output() as captured:
        display.clear_output(wait=True)
        display.display(plt.gcf())
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Number of games")
    plt.ylabel("Score")
    plt.plot(scores, alpha=0.8)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], f"{mean_scores[-1]:.2f}")
    sns.despine()
