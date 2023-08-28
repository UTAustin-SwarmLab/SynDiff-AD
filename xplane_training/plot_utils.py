import numpy as np
import matplotlib
import io

import matplotlib.pyplot as plt
import PIL.Image
from IPython import embed
from matplotlib.patches import Ellipse
from torchvision.transforms import ToTensor

matplotlib.use("Agg")


def basic_plot_ts(
    ts_vector, ts_vector_2, plot_file, legend=None, title_str=None, ylabel=None, lw=3.0, ylim=None, xlabel="time"
):
    plt.plot(ts_vector, lw=lw)
    plt.plot(ts_vector_2, lw=lw)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if ylim:
        plt.ylim(ylim[0], ylim[1])

    plt.title(title_str)
    plt.legend(legend)
    plt.savefig(plot_file)
    plt.close()


def plot_images_and_waypoints(image, orig_path, predicted_path):
    orig_path = orig_path.reshape(int(orig_path.shape[0] / 2), 2)
    predicted_path = predicted_path.reshape(int(predicted_path.shape[0] / 2), 2)

    orig_path = orig_path.detach().numpy()
    predicted_path = predicted_path.detach().numpy()

    plt.figure()
    idxs = np.where(image == 1)
    plt.scatter(idxs[0], idxs[1], c="r", label="obstacles")
    plt.scatter(orig_path[:, 0], orig_path[:, 1], c="b", label="original waypoints")
    plt.scatter(predicted_path[:, 0], predicted_path[:, 1], c="g", label="predicted waypoints")
    plt.legend()
    plt.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    plt.close()
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    return image


def plot_scene_waypoint(scene, history, waypoints, pred_waypoints, obstacles):
    fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})
    ells = [Ellipse(xy=(ox, oy), width=2 * ow, height=2 * oh) for (ox, oy, ow, oh) in obstacles]

    for e in ells:
        ax.add_artist(e)

    ax.set_xlim(0, scene.shape[0])
    ax.set_ylim(0, scene.shape[1])

    ax.plot(waypoints[:, 0], waypoints[:, 1], "-or", label="Orig Waypoint")
    ax.plot(pred_waypoints[:, 0], pred_waypoints[:, 1], "-ob", label="Predicted Waypoint")
    ax.plot(history[:, 0], history[:, 1], "-og", label="History")

    ax.grid(True)
    ax.legend()

    return fig, ax


def plot_adversarial_objects(fig, ax, obstacles, idx, waypoints, end=10):
    ells = [Ellipse(xy=(ox, oy), width=2 * ow, height=2 * oh) for (ox, oy, ow, oh) in obstacles]

    cmap = matplotlib.cm.get_cmap("coolwarm")
    color = cmap(float(idx / float(end)))

    alpha = 0.5 if (idx == 0 or idx == end - 1) else 0.2

    for e in ells:
        ax.add_artist(e)
        e.set(alpha=alpha, facecolor=color)

    ax.plot(waypoints[0, :], waypoints[1, :], "-o", c=color, alpha=alpha)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax.grid(True)

    return fig, ax


def plot_scene_detections(scene, obstacles, pred_obs):
    from IPython import embed

    embed()

    fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})
    ells = [Ellipse(xy=(ox, oy), width=2 * ow, height=2 * oh) for (ox, oy, ow, oh) in obstacles]
    pred_ells = [Ellipse(xy=(ox, oy), width=2 * ow, height=2 * oh) for (ox, oy, ow, oh) in pred_obs]

    for e in ells:
        ax.add_artist(e)
        e.set(alpha=0.5, facecolor="g", label="Orig")
    for e in pred_ells:
        ax.add_artist(e)
        e.set(alpha=0.5, facecolor="b", label="Pred")

    for i in range(scene.shape[0]):
        for j in range(scene.shape[1]):
            ax.scatter(i, j, alpha=min(scene[i, j], 0.1), c="r")

    ax.set_xlim(0, scene.shape[0])
    ax.set_ylim(0, scene.shape[1])

    ax.grid(True)
    ax.legend()

    return fig, ax


def plot_before_after(before, after):
    fig, ax = plt.subplots(figsize=(16, 9), nrows=1, ncols=2)

    before_ells = [Ellipse(xy=(ox, oy), width=2 * ow, height=2 * oh) for (ox, oy, ow, oh) in before[0]]
    after_ells = [Ellipse(xy=(ox, oy), width=2 * ow, height=2 * oh) for (ox, oy, ow, oh) in after[0]]

    after_ells_copy = [Ellipse(xy=(ox, oy), width=2 * ow, height=2 * oh) for (ox, oy, ow, oh) in after[0]]

    alpha = 0.2

    for e in before_ells:
        ax[0].add_patch(e)
        e.set(alpha=alpha, facecolor="b")

    for e in after_ells:
        ax[0].add_patch(e)
        e.set(alpha=alpha, facecolor="r")

    for e in after_ells_copy:
        ax[1].add_patch(e)
        e.set(alpha=alpha, facecolor="r")

    ax[0].plot(before[1][0, :], before[1][1, :], "-o", c="b", alpha=alpha)
    ax[0].title.set_text("Original Waypoints on Adversarial Scene")

    ax[1].plot(after[1][0, :], after[1][1, :], "-o", c="r", alpha=alpha)
    ax[1].title.set_text("New Waypoints on Adversarial Scene")

    for axes in ax:
        axes.set_xlim(0, 100)
        axes.set_ylim(0, 100)

        axes.grid(True)

    return fig
