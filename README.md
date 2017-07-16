# RAPT: A Python framework for tracing relativistic test particles in general electromagnetic fields

RAPT (Rice Adaptive Particle Tracer) is an object-oriented package for simulating trajectories of charged particles under electric and magnetic fields. It is intended to be used for plasma physics research and education.

RAPT has the following features that make it a generally useful:

* Relativistic.
* Complete particle, guiding center, and bounce-averaged trajectories.
* Automatic switching between particle and guiding-center trajectories for efficiency.
* Accurate, state-of-the-art equations for the guiding center.
* Static or time-dependent fields.
* Easy to extend with new field models.
* Field-line tracing.
* Utilites to evaluate particle-related or field-line related quantities.
* Modular, object-oriented design for easy maintenance and extension.

## Requirements

## Installation

## Examples
Jupyter notebooks illustrating RAPT with several specific problems.

* [Simulating a proton trajectory in the Earth's magnetic dipole field](https://github.com/mkozturk/rapt/blob/master/examples/Using%20the%20Particle%20class.ipynb) Illustrates the usage of the `Particle` class.

* [Electron guiding-center orbits under double-dipole field model](https://github.com/mkozturk/rapt/blob/master/examples/GuidingCenter%20Example%20-%20Electron%20under%20double%20dipole%20field.ipynb) Illustrates the usage of the `GuidingCenter` class.

* [Speiser orbits](https://github.com/mkozturk/rapt/blob/master/examples/Adaptive%20Example%20-%20Speiser%20orbits.ipynb) Illustrates the usage of the `Adaptive` class.

* [Drift-shell splitting: Bounce-averaged proton orbits](https://github.com/mkozturk/rapt/blob/master/examples/BounceCenter%20example%20-%20Drift-shell%20splitting.ipynb). Illustrates the usage of `BounceCenter` class.

* [Plotting field lines](https://github.com/mkozturk/rapt/blob/master/examples/Field%20line%20example.ipynb). How to visualize field lines in a given field model; plotting the field strength along a field line; getting the critical points (optima of field strength) on the field line. Illustrates the `Fieldline` class.

* [Creating new analytic field models](https://github.com/mkozturk/rapt/blob/master/examples/Creating%20new%20fields.ipynb) Illustrates generating new field models where the electric and magnetic fields have explicit mathematical formulas.

* 


