# Reservoir Management Optimization

## Overview

This Python project optimizes water release schedules for the Hirakud Dam, balancing three objectives: minimizing irrigation deficit, maximizing hydropower generation, and minimizing industrial water supply. It uses the DEAP library for evolutionary algorithms, implementing both single-objective (weighted sum) and multi-objective (NSGA-II) optimization. The model respects constraints such as reservoir storage limits, canal capacity, and flood control requirements, and includes visualizations to analyze results.

## Features

- **Single-Objective Optimization**: Uses a weighted sum approach with customizable weights for irrigation, hydropower, and industrial objectives.
- **Multi-Objective Optimization**: Generates a Pareto front using NSGA-II to explore trade-offs between objectives.
- **Constraints**: Enforces minimum/maximum storage, canal capacity, power generation limits, and flood storage requirements.
- **Visualizations**:
  - Monthly releases (irrigation, power, industrial), storage trajectories, hydropower, and spill.
  - Comparison plots across different weight combinations.
  - 3D Pareto front showing trade-offs between objectives.
- **Output Analysis**: Includes a summary table comparing objective values across weight sets.

## Requirements

- **Python**: 3.6 or higher
- **Libraries**:
  - `numpy`: For numerical computations
  - `matplotlib`: For plotting results
  - `deap`: For evolutionary algorithms

## Installation

1. **Install Python**: Download and install Python 3.6+ from [python.org](https://www.python.org/downloads/).
2. **Install Dependencies**: Use pip to install required libraries:
   pip install numpy matplotlib deap
