#!/usr/bin/env python3

import os
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import animation_tools as tools  # noqa: E402


class ChangeVisualizer(object):
    def __init__(self):

        self.frame_dir = "frames"
        self.output_dir = "output"
        try:
            os.mkdir(self.frame_dir)
        except Exception:
            pass
        try:
            os.mkdir(self.output_dir)
        except Exception:
            pass

        weight_min = .85
        weight_max = 1
        # slopes on the x-axis
        self.xmin, self.xmax = (weight_min, weight_max)
        # intercepts on the y-axis
        self.ymin, self.ymax = (weight_min, weight_max)
        # loss on the z-axis
        self.zmin, self.zmax = (0, .02)

        self.marker_large = 80
        self.marker_small = 8

        self.azim_min = -80
        self.azim_max = -35

        self.generate_data(n_pts=15)
        self.initialize_loss_function()

        fig = self.initialize_plot()
        self.scatter_ax = fig.add_subplot(1, 2, 1)
        self.loss_ax = fig.add_subplot(1, 2, 2, projection='3d')

        # Plot data only
        self.scatter(self.scatter_ax)
        # Plot loss function
        self.loss_surface()
        self.finalize_plot(filename="mnm_count_by_weight.png")

    def initialize_plot(self) -> Any:
        """
        Returns a Figure with two subplots.
        """
        fig = plt.figure(num=9334, figsize=(8, 4.5))
        fig.clf()
        plt.style.use("dark_background")
        return fig

    def generate_data(self, n_pts: int = None) -> np.ndarray:
        """
        Create the package weight and mnm count data.

        n_pts: The number of data points to simulate.

        Creates
        -------
        self.data: ndarray[float, float]
            Synthetic raw data.
            Rows are package weights.
            Columns are mnm counts.
        """
        np.random.seed(10)

        nominal_weight_before = .912  # grams
        nominal_weight_deviation_before = .01  # grams
        nominal_weight_after = .935  # grams
        nominal_weight_deviation_after = .015  # grams
        i_shift_day = int(n_pts * .5)

        day_interval = 7
        day_intervals = np.ceil(
            np.random.poisson(day_interval, n_pts)).astype(np.int)
        days = np.cumsum(day_intervals) - day_intervals[0]

        listed_data = []
        for i_day in range(n_pts):
            if i_day < i_shift_day:
                nominal_weight = nominal_weight_before
                nominal_weight_deviation = nominal_weight_deviation_before
            else:
                nominal_weight = nominal_weight_after
                nominal_weight_deviation = nominal_weight_deviation_after

            mnm_weight = np.random.normal(
                loc=nominal_weight,
                scale=nominal_weight_deviation,
            )

            listed_data.append([days[i_day], mnm_weight])

        self.data = np.array(listed_data)
        return

    def initialize_loss_function(
        self,
        shift_day: float = 27,
    ) -> None:
        """
        For a reasonable range of parameters, calculate the loss for all
        combinations.

        shift_day: The day on which the change was made.
        """
        weight_step = .001
        self.weights_before = np.arange(self.xmin, self.xmax, weight_step)
        self.weights_after = np.arange(self.ymin, self.ymax, weight_step)
        self.loss_function = np.zeros((
            self.weights_before.size, self.weights_after.size))
        for i_before, weight_before in enumerate(self.weights_before):
            for i_after, weight_after in enumerate(self.weights_after):
                loss = self.calculate_loss(
                    weight_before, weight_after, shift_day)
                self.loss_function[i_before, i_after] = loss
        return

    def calculate_loss(
        self,
        weight_before: float = None,
        weight_after: float = None,
        shift_day: float = None,
    ) -> float:
        """
        Find the loss for a given line.
        """
        x_days = self.data[:, 0]
        y_weights_actual = self.data[:, 1]
        y_weights_predicted = weight_after * np.ones(y_weights_actual.size)
        y_weights_predicted[x_days < shift_day] = weight_before
        y_weights_deviation = y_weights_actual - y_weights_predicted
        loss = np.sum(y_weights_deviation ** 2)
        loss_ceiling = self.zmax
        loss = np.minimum(loss, loss_ceiling)
        return loss

    def scatter(self, ax: Any = None) -> None:
        """
        Turn the data into a pretty scatter plot.

        Parameters
        -------
        Axis: The pyplot Axis the will contain the scatter plot.
        """
        ax.plot(
            self.data[:, 0],
            self.data[:, 1],
            linestyle="none",
            marker='.',
            markersize=self.marker_small,
        )
        ax.set_xlabel("Day")
        ax.set_ylabel("M&M weights (g)")
        ax.set_title("M&M weights over time")
        return

    def loss_surface(self, azim: float = -65, elev: float = 6) -> None:
        """
        Calculate the 3D surface of the loss function.
        """
        n_after, n_before = self.loss_function.shape
        # Build out the full 2D arrays for slopes and intercepts.
        after_arr = np.repeat(
            self.weights_after[:, np.newaxis], n_after, axis=1)
        before_arr = np.repeat(
            self.weights_before[:, np.newaxis], n_before, axis=1).T

        self.loss_ax.cla()
        self.loss_ax.view_init(elev=elev, azim=azim)
        self.loss_ax.contour(
            before_arr,
            after_arr,
            self.loss_function,
            4,
            offset=self.zmin,
            zdir="z",
            cmap=cm.Oranges,
            vmax=self.zmax,
            vmin=self.zmin,
        )

        self.loss_ax.plot_surface(
            before_arr,
            after_arr,
            self.loss_function,
            rcount=250,
            ccount=250,
            cmap=cm.YlGn,
            alpha=1,
            linewidth=0,
            vmax=self.zmax,
            vmin=self.zmin,
            # antialiased=False,
            antialiased=True,
        )
        self.loss_ax.set_xlabel("Weights before (g)")
        self.loss_ax.set_ylabel("after (g)")
        self.loss_ax.set_zlabel("Loss")
        self.loss_ax.set_title("Loss function")
        return

    def finalize_plot(
        self,
        dpi: int = 300,
        filename: str = None,
    ) -> None:
        """
        Take the image and save it to a file.

        dpi: Rendered dots per inch.
        """
        self.scatter_ax.set_xlim(
            np.min(self.data[:, 0]) - 1,
            np.max(self.data[:, 0]) + 1)
        self.scatter_ax.set_ylim(self.ymin, self.ymax)
        self.loss_ax.set_xlim(self.xmin, self.xmax)
        self.loss_ax.set_ylim(self.ymin, self.ymax)
        self.loss_ax.set_zlim(self.zmin, self.zmax)
        plt.savefig(filename, dpi=dpi)
        return

    def sweep_before(
        self,
        before_start: float = None,
        before_stop: float = None,
        weight_after: float = .93,
        shift_day: float = 56,
        duration: float = 16,
    ) -> None:
        """
        Animate a sweep through the values for the weight before.
        """
        if before_start is None:
            before_start = self.xmin
        if before_stop is None:
            before_stop = self.xmax
        clear_pngs(self.frame_dir)
        weights_before = tools.make_trajectory(
            duration=duration,
            waypoints=[
               before_start + 0 * (before_stop - before_start),
               before_start + 1 * (before_stop - before_start),
               before_start + .33 * (before_stop - before_start),
               before_start + .66 * (before_stop - before_start),
               before_start + .4 * (before_stop - before_start),
               before_start + .6 * (before_stop - before_start),
               before_start + .45 * (before_stop - before_start),
               before_start + .55 * (before_stop - before_start),
            ],
        )

        azimuths = tools.make_trajectory(
            duration=duration,
            max_submovement_duration=2,
            waypoints=[self.azim_min, self.azim_max],
        )

        frame_base = 10000
        self.initialize_loss_function(shift_day=shift_day)
        for i_frame, weight_before in enumerate(weights_before):
            frame_number = frame_base + i_frame

            self.render_frame_with_line(
                weight_before=weight_before,
                weight_after=weight_after,
                shift_day=shift_day,
                frame_number=frame_number,
            )

            self.loss_surface(azim=azimuths[i_frame])
            self.loss_ax.scatter(
                weight_before, weight_after,
                self.zmin - (self.zmax - self.zmin) * .01,
                color="blue",
                marker=".",
                s=self.marker_large,
            )

            filename = "mnm_weight_by_time_" + str(int(frame_number)) + ".png"
            pathname = os.path.join(self.frame_dir, filename)
            self.finalize_plot(filename=pathname)

        tools.render_movie(
            filename="mnm_before_weights_sweep.mp4",
            frame_dirname=self.frame_dir,
            output_dirname=self.output_dir,
        )
        return

    def sweep_after(
        self,
        after_start: float = None,
        after_stop: float = None,
        weight_before: float = .912,
        shift_day: float = 56,
        duration: float = 16,
    ) -> None:
        """
        Animate a sweep through the values for the weight after.
        """
        if after_start is None:
            after_start = self.xmin
        if after_stop is None:
            after_stop = self.xmax
        clear_pngs(self.frame_dir)
        weights_after = tools.make_trajectory(
            duration=duration,
            reverse=False,
            waypoints=[
               after_start + 0 * (after_stop - after_start),
               after_start + 1 * (after_stop - after_start),
               after_start + .33 * (after_stop - after_start),
               after_start + .66 * (after_stop - after_start),
               after_start + .4 * (after_stop - after_start),
               after_start + .6 * (after_stop - after_start),
               after_start + .45 * (after_stop - after_start),
               after_start + .55 * (after_stop - after_start),
            ],
        )

        azimuths = tools.make_trajectory(
            duration=duration,
            max_submovement_duration=2,
            reverse=False,
            waypoints=[self.azim_max, self.azim_min],
        )

        frame_base = 10000
        self.initialize_loss_function(shift_day=shift_day)
        for i_frame, weight_after in enumerate(weights_after):
            frame_number = frame_base + i_frame

            self.render_frame_with_line(
                weight_before=weight_before,
                weight_after=weight_after,
                shift_day=shift_day,
                frame_number=frame_number,
            )

            self.loss_surface(azim=azimuths[i_frame])
            self.loss_ax.scatter(
                weight_before, weight_after,
                self.zmin - (self.zmax - self.zmin) * .01,
                color="blue",
                marker=".",
                s=self.marker_large,
            )

            filename = "mnm_weight_by_time_" + str(int(frame_number)) + ".png"
            pathname = os.path.join(self.frame_dir, filename)
            self.finalize_plot(filename=pathname)

        tools.render_movie(
            filename="mnm_after_weights_sweep.mp4",
            frame_dirname=self.frame_dir,
            output_dirname=self.output_dir,
        )
        return

    def sweep_shift(
        self,
        shift_start: float = None,
        shift_stop: float = None,
        weight_before: float = .912,
        weight_after: float = .932,
        duration: float = 32,
    ) -> None:
        """
        Animate a sweep through the values for the weight after.
        """
        if shift_start is None:
            shift_start = np.min(self.data[:, 0]) + 1
        if shift_stop is None:
            shift_stop = np.max(self.data[:, 0]) - 1
        clear_pngs(self.frame_dir)
        shifts = tools.make_trajectory(
            duration=duration,
            waypoints=[
               shift_start + 0 * (shift_stop - shift_start),
               shift_start + 1 * (shift_stop - shift_start),
               shift_start + .33 * (shift_stop - shift_start),
               shift_start + .66 * (shift_stop - shift_start),
               shift_start + .4 * (shift_stop - shift_start),
               shift_start + .6 * (shift_stop - shift_start),
               shift_start + .45 * (shift_stop - shift_start),
               shift_start + .55 * (shift_stop - shift_start),
            ],
        )

        azimuths = tools.make_trajectory(
            duration=duration,
            max_submovement_duration=2,
            waypoints=[self.azim_min, self.azim_max],
        )

        frame_base = 10000
        for i_frame, shift_day in enumerate(shifts):
            frame_number = frame_base + i_frame

            self.render_frame_with_line(
                weight_before=weight_before,
                weight_after=weight_after,
                shift_day=shift_day,
                frame_number=frame_number,
            )

            self.initialize_loss_function(shift_day=shift_day)
            self.loss_surface(azim=azimuths[i_frame])
            self.loss_ax.scatter(
                weight_before, weight_after,
                self.zmin - (self.zmax - self.zmin) * .01,
                color="blue",
                marker=".",
                s=self.marker_large,
            )

            filename = "mnm_weight_by_time_" + str(int(frame_number)) + ".png"
            pathname = os.path.join(self.frame_dir, filename)
            self.finalize_plot(filename=pathname)

        filename = "mnm_shift_sweep.mp4"
        tools.render_movie(
            filename=filename,
            frame_dirname=self.frame_dir,
            output_dirname=self.output_dir,
        )

        tools.convert_to_gif(
            filename=filename,
            dirname=self.output_dir,
        )
        return

    def render_frame_with_line(
        self,
        weight_before: float = None,
        weight_after: float = None,
        shift_day: float = None,
        frame_number: int = 9999,
    ) -> None:
        """
        Plot the data and an instance of the model.
        """
        x_counts_before = [np.min(self.data[:, 0]), shift_day]
        y_weights_before = [weight_before, weight_before]
        x_counts_after = [shift_day, np.max(self.data[:, 0])]
        y_weights_after = [weight_after, weight_after]
        x_shift = [shift_day, shift_day]
        y_shift = [self.ymin, self.ymax]
        loss = self.calculate_loss(
            weight_before=weight_before,
            weight_after=weight_after,
            shift_day=shift_day,
        )

        self.scatter_ax.cla()
        self.scatter(self.scatter_ax)
        self.scatter_ax.plot(x_counts_before, y_weights_before)
        self.scatter_ax.plot(x_counts_after, y_weights_after)
        self.scatter_ax.plot(x_shift, y_shift, lw=.5)
        xf_left = .1
        yf_bottom = .7
        yf_delta = .05
        self.scatter_ax.text(
            self.xmin + (self.xmax - self.xmin) * xf_left,
            self.ymin + (self.ymax - self.ymin) * (yf_bottom + 3 * yf_delta),
            "weight before = {0:.3f} g".format(weight_before),
        )
        self.scatter_ax.text(
            self.xmin + (self.xmax - self.xmin) * xf_left,
            self.ymin + (self.ymax - self.ymin) * (yf_bottom + 2 * yf_delta),
            "weight after = {0:.3f} g".format(weight_after),
        )
        self.scatter_ax.text(
            self.xmin + (self.xmax - self.xmin) * xf_left,
            self.ymin + (self.ymax - self.ymin) * (yf_bottom + yf_delta),
            "shifted on day {0:.1f}".format(shift_day),
        )
        self.scatter_ax.text(
            self.xmin + (self.xmax - self.xmin) * xf_left,
            self.ymin + (self.ymax - self.ymin) * yf_bottom,
            "loss = {0:.4f}".format(loss),
        )
        return None


def clear_pngs(dirname: str = None) -> None:
    for filename in os.listdir(dirname):
        if ".png" in filename:
            os.remove(os.path.join(dirname, filename))
    return


def main():
    # viz = ChangeVisualizer()

    # viz.sweep_before(before_start=.851, before_stop=.975)
    # viz.sweep_after(after_start=.871, after_stop=.999)
    # viz.sweep_shift()
    filenames = [
        "mnm_before_weights_sweep.mp4",
        "mnm_after_weights_sweep.mp4",
        # "mnm_shift_sweep.mp4",
    ]
    for filename in filenames:
        tools.convert_to_gif(
            filename=filename,
            dirname="output",
        )
    return


if __name__ == "__main__":
    main()
