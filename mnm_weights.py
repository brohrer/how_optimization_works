#!/usr/bin/env python3

import os
import subprocess
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import animation_tools as tools  # noqa: E402


class WeightVisualizer(object):
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

        # slopes on the x-axis
        self.xmin, self.xmax = (.6, 1.2)
        # intercepts on the y-axis
        self.ymin, self.ymax = (-10, 25)
        # loss on the z-axis
        self.zmin, self.zmax, self.plot_zmax = (10, 100, 100)

        self.marker_large = 100
        self.marker_small = 10

        self.azim_min = -80
        self.azim_max = -35

        self.generate_data()
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

    def generate_data(self, n_pts: int = 15) -> np.ndarray:
        """
        Create the package weight and mnm count data.

        n_pts: The number of data points to simulate.

        Returns
        -------
        data: ndarray[float, float]
            Synthetic raw data.
            Rows are package weights.
            Columns are mnm counts.
        """
        np.random.seed(17)

        nominal_weight = .915  # grams
        nominal_weight_deviation = .02  # grams
        nominal_count = 55.5
        nominal_count_deviation = 3.5
        paper_weight = 4.2  # grams

        listed_data = []
        for _ in range(n_pts):
            mnm_count = int(np.random.normal(
                loc=nominal_count,
                scale=nominal_count_deviation,
            ))
            mnm_weight = np.random.normal(
                loc=nominal_weight,
                scale=nominal_weight_deviation,
            )
            package_weight = mnm_count * mnm_weight + paper_weight

            listed_data.append([package_weight, mnm_count])

        self.data = np.array(listed_data)
        return

    def initialize_loss_function(
        self,
        intercept_step: float = .1,
        slope_step: float = .005,
    ) -> None:
        """
        For a reasonable range of parameters, calculate the loss for all
        combinations.

        Creates
        -------
        slopes: The array of slopes used to calculate the loss function.
            These correspond to columns.
        intercepts: The array of intercepts used to calculate
            the loss function. These correspond to rows.
        loss_function: The value of the loss at each of the corresponding
            slopes and intercepts.
        """
        self.slopes = np.arange(self.xmin, self.xmax, slope_step)
        self.intercepts = np.arange(self.ymin, self.ymax, intercept_step)
        self.loss_function = np.zeros((self.intercepts.size, self.slopes.size))
        for i_intercept, intercept in enumerate(self.intercepts):
            for i_slope, slope in enumerate(self.slopes):
                loss = self.calculate_loss(intercept, slope)
                self.loss_function[i_intercept, i_slope] = loss
        return

    def calculate_loss(
        self,
        intercept: float = 4,
        slope: float = 1,
    ) -> float:
        """
        Find the loss for a given line.
        """
        x_counts_actual = self.data[:, 0]
        y_weights_actual = self.data[:, 1]
        y_weights_predicted = intercept + slope * x_counts_actual
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
        ax.set_xlabel("M&M counts")
        ax.set_ylabel("Package weight (g)")
        ax.set_title("M&M weights and candy counts")
        return

    def loss_surface(self, azim: float = -65, elev: float = 10) -> None:
        """
        Calculate the 3D surface of the loss function.
        """
        n_intercepts, n_slopes = self.loss_function.shape
        # Build out the full 2D arrays for slopes and intercepts.
        intercepts_arr = np.repeat(
            self.intercepts[:, np.newaxis], n_slopes, axis=1)
        slopes_arr = np.repeat(
            self.slopes[:, np.newaxis], n_intercepts, axis=1).T

        chopped_loss = self.loss_function.copy()
        # i_chop = np.where(chopped_loss >= self.plot_zmax)
        # chopped_loss[i_chop] = np.nan
        # intercepts_arr[i_chop] = np.nan
        # slopes_arr[i_chop] = np.nan

        self.loss_ax.cla()
        self.loss_ax.view_init(elev=elev, azim=azim)
        self.loss_ax.contour(
            slopes_arr,
            intercepts_arr,
            chopped_loss,
            2,
            offset=self.zmin,
            zdir="z",
            cmap=cm.Oranges,
            vmax=self.zmax,
            vmin=self.zmin,
        )

        self.loss_ax.plot_surface(
            slopes_arr,
            intercepts_arr,
            chopped_loss,
            # self.loss_function,
            rcount=250,
            ccount=250,
            cmap=cm.YlGn,
            alpha=.5,
            linewidth=0,
            vmax=self.zmax,
            vmin=self.zmin,
            antialiased=False,
        )
        self.loss_ax.set_xlabel("Slope")
        self.loss_ax.set_ylabel("Intercept")
        self.loss_ax.set_zlabel("Loss")
        self.loss_ax.set_title("Loss function for linear model")
        return

    def finalize_plot(
        self,
        dpi: int =150,
        filename: str = None,
    ) -> None:
        """
        Take the image and save it to a file.

        dpi: Rendered dots per inch.
        """
        self.scatter_ax.set_xlim(47, 66)
        self.scatter_ax.set_ylim(45, 75)
        self.loss_ax.set_xlim(self.xmin, self.xmax)
        self.loss_ax.set_ylim(self.ymin, self.ymax)
        self.loss_ax.set_zlim(self.zmin, self.plot_zmax)
        plt.savefig(filename, dpi=dpi)
        return

    def sweep_intercept(
        self,
        intercept_start: float = None,
        intercept_stop: float = None,
        duration: float = 9,
    ) -> None:
        """
        Animate a sweep through the values for the intercept.
        """
        if intercept_start is None:
            intercept_start = self.ymin
        if intercept_stop is None:
            intercept_stop = self.ymax
        clear_pngs(self.frame_dir)
        intercepts = tools.make_trajectory(
            duration=duration,
            reverse=True,
            waypoints=[
                intercept_start, intercept_stop,
                intercept_start, intercept_stop,
                intercept_start, intercept_stop,
            ],
        )

        azimuths = tools.make_trajectory(
            duration=duration,
            max_submovement_duration=2,
            reverse=True,
            waypoints=[self.azim_min, self.azim_max],
        )

        frame_base = 10000
        slope = 1
        for i_frame, intercept in enumerate(intercepts):
            frame_number = frame_base + i_frame

            self.render_frame_with_line(
                intercept=intercept,
                slope=slope,
                frame_number=frame_number,
            )

            self.loss_surface(azim=azimuths[i_frame])
            self.loss_ax.scatter(
                slope, intercept, self.zmin + 1,
                color="blue",
                marker=".",
                s=self.marker_large,
            )

            filename = "mnm_count_by_weight_" + str(int(frame_number)) + ".png"
            pathname = os.path.join(self.frame_dir, filename)
            self.finalize_plot(filename=pathname)

        render_movie(
            filename="mnm_weights_intercept_sweep.mp4",
            frame_dirname=self.frame_dir,
            output_dirname=self.output_dir,
        )
        return

    def sweep_slope(
        self,
        slope_start: float = None,
        slope_stop: float = None,
        duration: float = 9,
    ) -> None:
        """
        Animate a sweep through the values for the slope.
        """
        if slope_start is None:
            slope_start = self.xmin
        if slope_stop is None:
            slope_stop = self.xmax
        clear_pngs(self.frame_dir)
        slopes = tools.make_trajectory(
            duration=duration,
            reverse=True,
            waypoints=[
                slope_start, slope_stop,
                slope_start, slope_stop,
                slope_start, slope_stop,
            ],
        )

        azimuths = tools.make_trajectory(
            duration=duration,
            max_submovement_duration=2,
            reverse=True,
            waypoints=[self.azim_min, self.azim_max],
        )

        frame_base = 10000
        for i_frame, slope in enumerate(slopes):
            frame_number = frame_base + i_frame
            # This keeps the line passing through the middle of the data.
            intercept = 50 * (1 - slope)

            self.render_frame_with_line(
                intercept=intercept,
                slope=slope,
                frame_number=frame_number,
            )

            self.loss_surface(azim=azimuths[i_frame])
            self.loss_ax.scatter(
                slope, intercept, self.zmin + 1,
                color="blue",
                marker=".",
                s=self.marker_large,
            )

            filename = "mnm_count_by_weight_" + str(int(frame_number)) + ".png"
            pathname = os.path.join(self.frame_dir, filename)
            self.finalize_plot(filename=pathname)

        render_movie(
            filename="mnm_weights_slope_sweep.mp4",
            frame_dirname=self.frame_dir,
            output_dirname=self.output_dir,
        )
        return

    def sweep_intercept_slope(
        self,
        intercept_min: float = None,
        intercept_max: float = None,
        slope_min: float = None,
        slope_max: float = None,
        duration: float = 9,
        n_waypoints: int = 10,
    ) -> None:
        """
        Animate a sweep through the values for the slope.
        """
        if intercept_min is None:
            intercept_min = self.ymin
        if intercept_max is None:
            intercept_max = self.ymax
        if slope_min is None:
            slope_min = self.xmin
        if slope_max is None:
            slope_max = self.xmax
        clear_pngs(self.frame_dir)
        slope_waypoints = []
        intercept_waypoints = []
        for _ in range(n_waypoints):
            intercept_waypoints.append(
                np.random.sample() * (intercept_max - intercept_min)
                + intercept_min)
            slope_waypoints.append(
                np.random.sample() * (slope_max - slope_min)
                + slope_min)

        slopes = tools.make_trajectory(
            duration=duration,
            loop=True,
            waypoints=slope_waypoints,
        )
        intercepts = tools.make_trajectory(
            duration=duration,
            loop=True,
            waypoints=intercept_waypoints,
        )

        azimuths = tools.make_trajectory(
            duration=duration,
            max_submovement_duration=2,
            reverse=True,
            waypoints=[self.azim_min, self.azim_max],
        )

        frame_base = 10000
        for i_frame, slope in enumerate(slopes):
            local_intercept = intercepts[i_frame]
            frame_number = frame_base + i_frame
            # This keeps the line passing through the middle of the data.
            intercept = local_intercept + 56 * (1 - slope)

            self.render_frame_with_line(
                intercept=intercept,
                slope=slope,
                frame_number=frame_number,
            )

            self.loss_surface(azim=azimuths[i_frame])
            self.loss_ax.scatter(
                slope, intercept, self.zmin + 1,
                color="blue",
                marker=".",
                s=self.marker_large,
            )

            filename = "mnm_count_by_weight_" + str(int(frame_number)) + ".png"
            pathname = os.path.join(self.frame_dir, filename)
            self.finalize_plot(filename=pathname)

        render_movie(
            filename="mnm_weights_random_sweep.mp4",
            frame_dirname=self.frame_dir,
            output_dirname=self.output_dir,
        )
        return

    def render_frame_with_line(
        self,
        intercept: float = 4,
        slope: float =.94,
        frame_number: int =9999,
    ) -> None:
        """
        Plot the data and an instance of the linear model.
        """
        x_counts = np.linspace(48, 65)
        y_weights = intercept + slope * x_counts
        loss = self.calculate_loss(intercept, slope)

        self.scatter_ax.cla()
        self.scatter(self.scatter_ax)
        self.scatter_ax.plot(x_counts, y_weights)
        self.scatter_ax.text(50, 70, "slope = {0:.2f}".format(slope))
        self.scatter_ax.text(50, 68, "intercept = {0:.2f}".format(intercept))
        self.scatter_ax.text(50, 66, "loss = {0}".format(int(loss)))
        return None


def render_movie(
    filename: str = None,
    frame_dirname: str = None,
    output_dirname: str = None,
) -> None:

    movie_path = os.path.join(output_dirname, filename)

    # Prepare the arguments for the call to FFmpeg.
    input_file_format = "*.png"
    input_file_pattern = os.path.join(frame_dirname, input_file_format)
    codec = "libx264"
    command = [
        "ffmpeg",
        "-pattern_type", "glob",
        "-i", input_file_pattern,
        "-y",
        "-c:v", codec,
        movie_path,
    ]
    print(" ".join(command))
    subprocess.call(command)
    return


def clear_pngs(dirname: str = None) -> None:
    for filename in os.listdir(dirname):
        if ".png" in filename:
            os.remove(os.path.join(dirname, filename))
    return


def main():
    viz = WeightVisualizer()

    # viz.sweep_intercept()
    # viz.sweep_slope()
    viz.sweep_intercept_slope(
        intercept_min=-2,
        intercept_max=5,
        slope_min=0,
        slope_max=2,
    )
    return


if __name__ == "__main__":
    main()
