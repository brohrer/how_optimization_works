import matplotlib
matplotlib.use("agg")
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

mnm_counts = [
    45,
    47,
    48,
    49, 49,
    51, 51, 51, 51,
    52, 52, 52,
    53, 53, 53, 53, 53,
    54, 54,
    55, 55, 55, 55, 55, 55, 55, 55,
    56, 56, 56, 56, 56,
    57, 57, 57, 57, 57, 57, 57,
    58, 58, 58,
    59, 59, 59, 59, 59, 59,
    60, 60, 60, 60,
    61, 61,
    63,
    64,
]


def point_loss(actual, estimate):
    return (actual - estimate) ** 2


def total_loss(actuals, estimate):
    loss = 0
    for actual in actuals:
        loss += point_loss(actual, estimate)
    return loss


def loss_curve(actuals):
    min_estimate = 40
    max_estimate = 70
    delta_estimate = .01
    estimates = np.arange(min_estimate, max_estimate, delta_estimate)
    losses = np.zeros(estimates.size)
    for i_estimate, estimate in enumerate(estimates):
        losses[i_estimate] = total_loss(actuals, estimate)
    return estimates, losses


def plot_losses(estimates, losses):
    plt.figure(figsize=(8, 4))
    plt.style.use("dark_background")
    plt.plot(estimates, losses)
    plt.xlabel("M&M count estimates")
    plt.ylabel("Loss")
    plt.savefig("mnm_losses.png")


if __name__ == "__main__":
    estimates, losses = loss_curve(mnm_counts)
    plot_losses(estimates, losses)
    best_estimate = estimates[np.argmin(losses)]
    print("Best estimate is {0:.2f} M&Ms per bag.".format(best_estimate))
    print("The mean value is {0:.2f} M&Ms per bag."
          .format(np.mean(np.array(mnm_counts))))
