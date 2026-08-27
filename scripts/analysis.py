#!/usr/bin/env python3

"""
@file   analysis.py
@brief  Plotting and analysis tool to determine IMU parameters.
@author Russell Buchanan
"""

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
from scipy.optimize import curve_fit


# Publication-oriented sans-serif defaults to match a sans-serif body font.
plt.rcParams.update({
	"font.family": "sans-serif",
	"font.sans-serif": ["Arimo", "Arial", "Helvetica", "DejaVu Sans"],
	"mathtext.fontset": "dejavusans",
	"font.size": 11,
	"axes.labelsize": 10.5,
	"axes.titlesize": 10.5,
	"axes.titleweight": "bold",
	"axes.linewidth": 0.8,
	"legend.fontsize": 9.5,
	"xtick.labelsize": 10.5,
	"ytick.labelsize": 10.5,
	"lines.linewidth": 1.4,
	"savefig.dpi": 600,
	"savefig.bbox": "tight",
	"pdf.fonttype": 42,
	"ps.fonttype": 42,
})


def style_log_axes(ax):
	"""Apply compact, publication-quality formatting to a log-log plot."""
	major_locator = LogLocator(base=10)
	minor_locator = LogLocator(base=10, subs=np.arange(2, 10) * 0.1)
	formatter = LogFormatterMathtext(base=10)

	ax.xaxis.set_major_locator(major_locator)
	ax.xaxis.set_minor_locator(minor_locator)
	ax.xaxis.set_major_formatter(formatter)
	ax.xaxis.set_minor_formatter(NullFormatter())
	ax.yaxis.set_major_locator(LogLocator(base=10))
	ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
	ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
	ax.yaxis.set_minor_formatter(NullFormatter())
	ax.tick_params(which="both", direction="in", top=True, right=True)
	ax.tick_params(which="major", length=5, width=0.8)
	ax.tick_params(which="minor", length=3, width=0.6)
	ax.grid(which="major", color="0.82", linewidth=0.6, linestyle="--")
	ax.grid(which="minor", visible=False)


def save_publication_figure(fig, basename):
	"""Save a high-resolution raster image and an editable vector image."""
	fig.savefig(basename + ".png")
	fig.savefig(basename + ".pdf")


def plot_allan_panel(ax, period, data, white_fits, random_walk_fits,
					 white_intercepts, random_walk_intercepts, title, ylabel):
	"""Draw one Allan-deviation panel for a combined vector figure."""
	colors = ("#0072B2", "#D55E00", "#009E73")
	linestyles = ("-", "--", "-.")
	axis_labels = (r"$x$", r"$y$", r"$z$")

	for axis in range(3):
		ax.loglog(period, data[:, axis], color=colors[axis],
				  linestyle=linestyles[axis], label=axis_labels[axis])
		ax.loglog(period, white_fits[axis](period), color="0.45", linestyle=":",
				  label="white-noise" if axis == 2 else None)
		ax.loglog(period, random_walk_fits[axis](period), color="#CC79A7",
				  linestyle=":", label="random-walk" if axis == 2 else None)
		ax.loglog(1.0, white_intercepts[axis], "o", color=colors[axis], markersize=7)
		ax.loglog(3.0, random_walk_intercepts[axis], "*", color=colors[axis], markersize=9)

	ax.set_title(title)
	ax.set_ylabel(ylabel)
	ax.set_xlabel(r"Period, $\tau$ ($\mathrm{s}$)")
	ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.7,
			  edgecolor="0.75", ncol=2, columnspacing=1.0, handlelength=1.8)
	style_log_axes(ax)


def line_func(x, m, b):
	"""
	Linear function to fit the data.

	Args:
		x (np.array): The x values to fit.
		m (float): The slope of the line.
		b (float): The y-intercept of the line.

	Returns:
		np.array: The y values of the line.
	"""
	return m * x + b

def get_intercept(x, y, m, b):
	"""
	Get the intercept of the line.

	Args:
		x (np.array): The x values to fit.
		y (np.array): The y values to fit.
		m (float): The slope of the line.
		b (float): The y-intercept of the line.

	Returns:
		float: The y-intercept of the line.
	"""

	logx = np.log(x)
	logy = np.log(y)
	coeffs, _ = curve_fit(line_func, logx, logy, bounds=([m, -np.inf], [m + 0.001, np.inf]))
	poly = np.poly1d(coeffs)
	yfit = lambda x: np.exp(poly(np.log(x)))

	return yfit(b), yfit


def generate_prediction(tau, q_quantization=0, q_white=0, q_bias_instability=0, q_walk=0, q_ramp=0):
	"""
	Generate a prediction of the Allan deviation.

	Args:
		tau (np.array): The time periods to predict.
		q_quantization (float): The quantization noise.
		q_white (float): The white noise.
		q_bias_instability (float): The bias instability.
		q_walk (float): The random walk.
		q_ramp (float): The ramp.
	"""
	n = len(tau)

	# Create the matrix A
	A = np.empty((n, 5))
	A[:, 0] = 3 / tau**2
	A[:, 1] = 1 / tau
	A[:, 2] = 2 * np.log(2) / np.pi
	A[:, 3] = tau / 3
	A[:, 4] = tau**2 / 2

	# Create the parameters vector
	params = np.array([q_quantization ** 2, q_white ** 2, q_bias_instability ** 2, q_walk ** 2, q_ramp ** 2])

	# Calculate the Allan deviation
	return np.sqrt(A.dot(params))

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data', metavar='STR', type=str, help='TUM data files to plot')
parser.add_argument('--config', metavar='STR', type=str, help='yaml config file')
parser.add_argument("--skip", type=int, default=1)
parser.add_argument("--output", type=str, default="imu.yaml")
args = parser.parse_args()

# Load config file if provided
rostopic = "/sensors/imu"
update_rate = 400.0
if args.config:
	import yaml #pip install pyyaml
	with open(args.config, "r") as stream:
		config = yaml.safe_load(stream)
	rostopic = config["imu_topic"]
	update_rate = config["imu_rate"]

# Assumes tum format
# Initialize arrays
period = np.array([])
acceleration = np.empty((0,3), float)
rotation_rate = np.empty((0,3), float)

# Read data from file
with open(args.data) as input_file:
	csv_reader = csv.reader(input_file, delimiter=' ')
	counter = 0

	for row in csv_reader:
		counter = counter + 1
		if (counter % args.skip != 0):
			continue
		# Data is in the format:
		# period, accel_x_allan_deviation, accel_y_allan_deviation, accel_z_allan_deviation, gyro_x_allan_deviation, gyro_y_allan_deviation, gyro_z_allan_deviation
		t = float(row[0])
		period = np.append(period, [t], axis=0)
		acceleration = np.append(acceleration, np.array([float(row[1]), float(row[2]), float(row[3])]).reshape(1,3), axis=0)
		rotation_rate = np.append(rotation_rate, np.array([float(row[4]), float(row[5]), float(row[6])]).reshape(1,3), axis=0)

# Calculate intercepts and minimums for accelerometer
white_noise_break_point = np.where(period == 10)[0][0]
random_rate_break_point = np.where(period == 10)[0][0]

# White Noise Intercept with a line of the form y = -0.5 * x + 1.0
accel_wn_intercept_x, xfit_wn = get_intercept(period[0:white_noise_break_point], acceleration[0:white_noise_break_point,0], -0.5, 1.0)
accel_wn_intercept_y, yfit_wn = get_intercept(period[0:white_noise_break_point], acceleration[0:white_noise_break_point,1], -0.5, 1.0)
accel_wn_intercept_z, zfit_wn = get_intercept(period[0:white_noise_break_point], acceleration[0:white_noise_break_point,2], -0.5, 1.0)

# Random Walk Intercept with a line of the form y = 0.5 * x + 3.0
accel_rr_intercept_x, xfit_rr = get_intercept(period, acceleration[:,0], 0.5, 3.0)
accel_rr_intercept_y, yfit_rr = get_intercept(period, acceleration[:,1], 0.5, 3.0)
accel_rr_intercept_z, zfit_rr = get_intercept(period, acceleration[:,2], 0.5, 3.0)

# Keep the accelerometer fits because the generic fit variables are reused below.
accel_white_fits = (xfit_wn, yfit_wn, zfit_wn)
accel_random_walk_fits = (xfit_rr, yfit_rr, zfit_rr)
accel_white_intercepts = (accel_wn_intercept_x, accel_wn_intercept_y, accel_wn_intercept_z)
accel_random_walk_intercepts = (accel_rr_intercept_x, accel_rr_intercept_y, accel_rr_intercept_z)

accel_min_x = np.amin(acceleration[:,0])
accel_min_y = np.amin(acceleration[:,1])
accel_min_z = np.amin(acceleration[:,2])

accel_min_x_index = np.argmin(acceleration[:,0])
accel_min_y_index = np.argmin(acceleration[:,1])
accel_min_z_index = np.argmin(acceleration[:,2])

# Use worst value among all axes x,y,z
worst_accel_white_noise = np.amax([accel_wn_intercept_x, accel_wn_intercept_y, accel_wn_intercept_z])
worst_accel_random_walk = np.amax([accel_rr_intercept_x, accel_rr_intercept_y, accel_rr_intercept_z])

# Write to yaml file
yaml_file = open(args.output, "w")
yaml_file.write("#Accelerometer\n")
yaml_file.write("accelerometer_noise_density: " + repr(worst_accel_white_noise) + " \n")
yaml_file.write("accelerometer_random_walk: " + repr(worst_accel_random_walk) + " \n")
yaml_file.write("\n")

print("ACCELEROMETER:")
print(f"X Velocity Random Walk: {accel_wn_intercept_x: .5f} m/s/sqrt(s) {accel_wn_intercept_x*60: .5f} m/s/sqrt(hr)")
print(f"Y Velocity Random Walk: {accel_wn_intercept_y: .5f} m/s/sqrt(s) {accel_wn_intercept_y*60: .5f} m/s/sqrt(hr)")
print(f"Z Velocity Random Walk: {accel_wn_intercept_z: .5f} m/s/sqrt(s) {accel_wn_intercept_z*60: .5f} m/s/sqrt(hr)")

print(f"X Bias Instability: {accel_min_x: .5f} m/s^2 {accel_min_x*3600*3600: .5f} m/hr^2")
print(f"Y Bias Instability: {accel_min_y: .5f} m/s^2 {accel_min_y*3600*3600: .5f} m/hr^2")
print(f"Z Bias Instability: {accel_min_z: .5f} m/s^2 {accel_min_z*3600*3600: .5f} m/hr^2")

print(f"X Accel Random Walk: {accel_rr_intercept_x: .5f} m/s^2/sqrt(s)")
print(f"Y Accel Random Walk: {accel_rr_intercept_y: .5f} m/s^2/sqrt(s)")
print(f"Z Accel Random Walk: {accel_rr_intercept_z: .5f} m/s^2/sqrt(s)")

average_acc_white_noise = (accel_wn_intercept_x + accel_wn_intercept_y + accel_wn_intercept_z) / 3
average_acc_bias_instability = (accel_min_x + accel_min_y + accel_min_z) / 3
average_acc_random_walk = (accel_rr_intercept_x + accel_rr_intercept_y + accel_rr_intercept_z) / 3

# Plot accelerometer data
dpi = 150
# Use a wider aspect ratio so the log-scale curves remain easy to distinguish
# when the figure is scaled to a single column in a two-column paper.
figsize = (5.0, 3.2)
fig1 = plt.figure(num="Acceleration", dpi=dpi, figsize=figsize)

plt.loglog(period, acceleration[:,0], color="#0072B2", linestyle="-", label=r"$x$")
plt.loglog(period, acceleration[:,1], color="#D55E00", linestyle="--", label=r"$y$")
plt.loglog(period, acceleration[:,2], color="#009E73", linestyle="-.", label=r"$z$")

plt.loglog(period, xfit_wn(period), color="0.45", linestyle=":")
plt.loglog(period, yfit_wn(period), color="0.45", linestyle=":")
plt.loglog(period, zfit_wn(period), color="0.45", linestyle=":", label="white-noise")

plt.loglog(period, xfit_rr(period), color="#CC79A7", linestyle=":")
plt.loglog(period, yfit_rr(period), color="#CC79A7", linestyle=":")
plt.loglog(period, zfit_rr(period), color="#CC79A7", linestyle=":", label="random-walk")

plt.loglog(1.0, accel_wn_intercept_x, "o", color="#0072B2", markersize=7)
plt.loglog(1.0, accel_wn_intercept_y, "o", color="#D55E00", markersize=7)
plt.loglog(1.0, accel_wn_intercept_z, "o", color="#009E73", markersize=7)

plt.loglog(3.0, accel_rr_intercept_x, "*", color="#0072B2", markersize=9)
plt.loglog(3.0, accel_rr_intercept_y, "*", color="#D55E00", markersize=9)
plt.loglog(3.0, accel_rr_intercept_z, "*", color="#009E73", markersize=9)

# Bias-instability markers (disabled for publication figures).
# plt.loglog(period[accel_min_x_index], accel_min_x, "^", color="#0072B2", markersize=7)
# plt.loglog(period[accel_min_y_index], accel_min_y, "^", color="#D55E00", markersize=7)
# plt.loglog(period[accel_min_z_index], accel_min_z, "^", color="#009E73", markersize=7)

# Combined fitted model (disabled for publication figures).
# fitted_model = generate_prediction(period, q_white=average_acc_white_noise,
# 									 q_bias_instability=average_acc_bias_instability,
# 									 q_walk=average_acc_random_walk)
# plt.loglog(period, fitted_model, "-k", linewidth=1.6, label="Model fit")

plt.title("Accelerometer")
plt.ylabel(r"Allan deviation, $\sigma_A(\tau)$ ($\mathrm{m\,s^{-2}}$)")
plt.xlabel(r"Period, $\tau$ ($\mathrm{s}$)")
plt.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.7,
		   edgecolor="0.75", ncol=2, columnspacing=1.0, handlelength=1.8)
style_log_axes(plt.gca())
plt.tight_layout()

plt.draw()
plt.pause(1)
w = plt.waitforbuttonpress(timeout=5)
save_publication_figure(fig1, "acceleration")
plt.close(fig1)

# Calculate intercepts and minimums for gyroscope
gyro_wn_intercept_x, xfit_wn = get_intercept(period[0:white_noise_break_point], rotation_rate[0:white_noise_break_point,0], -0.5, 1.0)
gyro_wn_intercept_y, yfit_wn = get_intercept(period[0:white_noise_break_point], rotation_rate[0:white_noise_break_point,1], -0.5, 1.0)
gyro_wn_intercept_z, zfit_wn = get_intercept(period[0:white_noise_break_point], rotation_rate[0:white_noise_break_point,2], -0.5, 1.0)

gyro_rr_intercept_x, xfit_rr = get_intercept(period, rotation_rate[:,0], 0.5, 3.0)
gyro_rr_intercept_y, yfit_rr = get_intercept(period, rotation_rate[:,1], 0.5, 3.0)
gyro_rr_intercept_z, zfit_rr = get_intercept(period, rotation_rate[:,2], 0.5, 3.0)

gyro_white_fits = (xfit_wn, yfit_wn, zfit_wn)
gyro_random_walk_fits = (xfit_rr, yfit_rr, zfit_rr)
gyro_white_intercepts = (gyro_wn_intercept_x, gyro_wn_intercept_y, gyro_wn_intercept_z)
gyro_random_walk_intercepts = (gyro_rr_intercept_x, gyro_rr_intercept_y, gyro_rr_intercept_z)

gyro_min_x = np.amin(rotation_rate[:,0])
gyro_min_y = np.amin(rotation_rate[:,1])
gyro_min_z = np.amin(rotation_rate[:,2])

gyro_min_x_index = np.argmin(rotation_rate[:,0])
gyro_min_y_index = np.argmin(rotation_rate[:,1])
gyro_min_z_index = np.argmin(rotation_rate[:,2])

# use worst value
worst_gyro_white_noise = np.amax([gyro_wn_intercept_x, gyro_wn_intercept_y, gyro_wn_intercept_z])
worst_gyro_random_walk = np.amax([gyro_rr_intercept_x, gyro_rr_intercept_y, gyro_rr_intercept_z])

# Write gyroscope parameters to yaml file
yaml_file.write("#Gyroscope\n")
# Convert back to radians here
yaml_file.write("gyroscope_noise_density: " + repr(worst_gyro_white_noise * np.pi / 180) + " \n")
yaml_file.write("gyroscope_random_walk: " + repr(worst_gyro_random_walk * np.pi / 180) + " \n")
yaml_file.write("\n")

print("GYROSCOPE:")
print(f"X Angle Random Walk: {gyro_wn_intercept_x: .5f} deg/sqrt(s) {gyro_wn_intercept_x * 60: .5f} deg/sqrt(hr)")
print(f"Y Angle Random Walk: {gyro_wn_intercept_y: .5f} deg/sqrt(s) {gyro_wn_intercept_y * 60: .5f} deg/sqrt(hr)")
print(f"Z Angle Random Walk: {gyro_wn_intercept_z: .5f} deg/sqrt(s) {gyro_wn_intercept_z * 60: .5f} deg/sqrt(hr)")

print(f"X Bias Instability: {gyro_min_x: .5f} deg/s {gyro_min_x*60*60: .5f} deg/hr")
print(f"Y Bias Instability: {gyro_min_y: .5f} deg/s {gyro_min_y*60*60: .5f} deg/hr")
print(f"Z Bias Instability: {gyro_min_z: .5f} deg/s {gyro_min_z*60*60: .5f} deg/hr")

print(f"X Rate Random Walk: {gyro_rr_intercept_x: .5f} deg/s/sqrt(s)")
print(f"Y Rate Random Walk: {gyro_rr_intercept_y: .5f} deg/s/sqrt(s)")
print(f"Z Rate Random Walk: {gyro_rr_intercept_z: .5f} deg/s/sqrt(s)")

average_gyro_white_noise = (gyro_wn_intercept_x + gyro_wn_intercept_y + gyro_wn_intercept_z) / 3
average_gyro_bias_instability = (gyro_min_x + gyro_min_y + gyro_min_z) / 3
average_gyro_random_walk = (gyro_rr_intercept_x + gyro_rr_intercept_y + gyro_rr_intercept_z) / 3

# Plot gyroscope data
fig2 = plt.figure(num="Gyro", dpi=dpi, figsize=figsize)

plt.loglog(period, rotation_rate[:,0], color="#0072B2", linestyle="-", label=r"$x$")
plt.loglog(period, rotation_rate[:,1], color="#D55E00", linestyle="--", label=r"$y$")
plt.loglog(period, rotation_rate[:,2], color="#009E73", linestyle="-.", label=r"$z$")

plt.loglog(period, xfit_wn(period), color="0.45", linestyle=":")
plt.loglog(period, yfit_wn(period), color="0.45", linestyle=":")
plt.loglog(period, zfit_wn(period), color="0.45", linestyle=":", label="white-noise")

plt.loglog(period, xfit_rr(period), color="#CC79A7", linestyle=":")
plt.loglog(period, yfit_rr(period), color="#CC79A7", linestyle=":")
plt.loglog(period, zfit_rr(period), color="#CC79A7", linestyle=":", label="random-walk")

plt.loglog(1.0, gyro_wn_intercept_x, "o", color="#0072B2", markersize=7)
plt.loglog(1.0, gyro_wn_intercept_y, "o", color="#D55E00", markersize=7)
plt.loglog(1.0, gyro_wn_intercept_z, "o", color="#009E73", markersize=7)

plt.loglog(3.0, gyro_rr_intercept_x, "*", color="#0072B2", markersize=9)
plt.loglog(3.0, gyro_rr_intercept_y, "*", color="#D55E00", markersize=9)
plt.loglog(3.0, gyro_rr_intercept_z, "*", color="#009E73", markersize=9)

# Bias-instability markers (disabled for publication figures).
# plt.loglog(period[gyro_min_x_index], gyro_min_x, "^", color="#0072B2", markersize=7)
# plt.loglog(period[gyro_min_y_index], gyro_min_y, "^", color="#D55E00", markersize=7)
# plt.loglog(period[gyro_min_z_index], gyro_min_z, "^", color="#009E73", markersize=7)

# Combined fitted model (disabled for publication figures).
# fitted_model = generate_prediction(period, q_white=average_gyro_white_noise,
# 									 q_bias_instability=average_gyro_bias_instability,
# 									 q_walk=average_gyro_random_walk)
# plt.loglog(period, fitted_model, "-k", linewidth=1.6, label="Model fit")

plt.title("Gyroscope")
plt.ylabel(r"Allan deviation, $\sigma_A(\tau)$ ($^\circ\!\mathrm{s^{-1}}$)")
plt.xlabel(r"Period, $\tau$ ($\mathrm{s}$)")
plt.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.7,
		   edgecolor="0.75", ncol=2, columnspacing=1.0, handlelength=1.8)
style_log_axes(plt.gca())
plt.tight_layout()

plt.draw()
plt.pause(1)
w = plt.waitforbuttonpress(timeout=5)
save_publication_figure(fig2, "gyro")
plt.close(fig2)

# Save vector-quality combined figures for alternative paper layouts.
accel_ylabel = r"Allan deviation, $\sigma_A(\tau)$ ($\mathrm{m\,s^{-2}}$)"
gyro_ylabel = r"Allan deviation, $\sigma_A(\tau)$ ($^\circ\!\mathrm{s^{-1}}$)"

fig_vertical, axes_vertical = plt.subplots(2, 1, dpi=dpi, figsize=(5.0, 6.4))
plot_allan_panel(axes_vertical[0], period, acceleration,
				 accel_white_fits, accel_random_walk_fits,
				 accel_white_intercepts, accel_random_walk_intercepts,
				 "Accelerometer", accel_ylabel)
plot_allan_panel(axes_vertical[1], period, rotation_rate,
				 gyro_white_fits, gyro_random_walk_fits,
				 gyro_white_intercepts, gyro_random_walk_intercepts,
				 "Gyroscope", gyro_ylabel)
fig_vertical.tight_layout()
fig_vertical.savefig("allan_acc_gyro_vertical.pdf")
plt.close(fig_vertical)

fig_horizontal, axes_horizontal = plt.subplots(1, 2, dpi=dpi, figsize=(10.0, 3.2))
plot_allan_panel(axes_horizontal[0], period, acceleration,
				 accel_white_fits, accel_random_walk_fits,
				 accel_white_intercepts, accel_random_walk_intercepts,
				 "Accelerometer", accel_ylabel)
plot_allan_panel(axes_horizontal[1], period, rotation_rate,
				 gyro_white_fits, gyro_random_walk_fits,
				 gyro_white_intercepts, gyro_random_walk_intercepts,
				 "Gyroscope", gyro_ylabel)
fig_horizontal.tight_layout(w_pad=2.0)
fig_horizontal.savefig("allan_acc_gyro_horizontal.pdf")
plt.close(fig_horizontal)

# Write rostopic and update rate to yaml file
if args.config:
	yaml_file.write("rostopic: " + repr(rostopic) + " \n")
	yaml_file.write("update_rate: " + repr(update_rate) + " \n")
else:
	yaml_file.write("rostopic: " + repr(rostopic) + " #Make sure this is correct\n")
	yaml_file.write("update_rate: " + repr(update_rate) + " #Make sure this is correct\n")
yaml_file.write("\n")
yaml_file.close()

print("Writing Kalibr imu.yaml file.")
print("Make sure to update the rostopic and rate in the file if a config file was not provided.")
