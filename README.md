# Small-Scale Impact Gardening Simulation Model

This repository contains the source code for the three-dimensional gardening simulation used in the paper: **Million-Year Scale Meteoroid Flux Homogeneity Across the Moon: Evidence from Chang’E6 Farside Samples and Gardening Simulations**

The code is modified from the open-source model published by **P. O’Brien et al.** in **[https://zenodo.org/records/4042167](https://zenodo.org/records/4042167)** and has been adapted to simulate small-scale regolith gardening at the **Chang’e6 (CE6)** landing site.

---

## Usage Instructions

### 1. Parameter Settings

Simulation parameters are defined in **params.py**:

```python
resolution = 0.01 # Spatial resolution (m) 
dt = 1e6 # Time step (year) 
grid_width = 2.0 # Physical width of the simulation grid (m) 
model_time = 2e7 # Total simulation time (year) 
n_particles_per_layer = 10 # Number of tracer particles per layer (rows × columns) 
save_dir = ".//Output//" # Output directory
```

In **Tracers.py**, the `build_tracers_group()` function defines:

* The number of layers (default: 100)
* The vertical range for each layer

For example, when `n_particles_per_layer = 10`, each layer contains **10 × 10 = 100** particles, and the entire simulation includes **10 × 10 × 100 = 10,000** particles in total.

---

### 2. Running the Simulation

Run the main script:

```bash
python main.py
```

Simulation results will be saved in the **Output/** directory.

The following output files will be generated:

| Filename                  | Description                                             |
| ------------------------- | ------------------------------------------------------- |
| `Craters_Information.csv` | Records the location and diameter of each impact crater |
| `t_line.npy`              | Timestamps of each impact event                         |
| `partical_depth.npy`      | Particle motion trajectories induced by impacts         |
| `Crater_Map.npy`          | Digital Elevation Model (DEM) of the simulated region   |

---

### 3. Visualization Tools

Several visualization scripts are provided in the **display/** folder:

| Script                  | Description                                                                      |
| ----------------------- | -------------------------------------------------------------------------------- |
| `draw_impact_dem.py`    | Visualizes crater formation sequence and DEM evolution                           |
| `draw_track.py`         | Plots particle motion paths and spatial distributions                            |
| `draw_sample_grains.py` | Displays particle exposure ages and source origins within a specific depth range |

A demonstration dataset can be downloaded from:

**[https://pan.baidu.com/s/1Egr4hF7ujG7UdC0brGh2fA?pwd=1234](https://pan.baidu.com/s/1Egr4hF7ujG7UdC0brGh2fA?pwd=1234)**

---

### 4. Exposure Age Correction

During visualization, the calculated exposure ages are divided by 4 before plotting. This adjustment accounts for shielding effects, as the exposure age recorded by each grain is approximately one-quarter of its actual surface residence time.

The factor of **1/4** was derived from 3D grain simulations implemented in **grain_3D.py**, which performs:

* Generation of 3D spherical surface
* Accumulation of exposure ages
* Random rotation of the 3D surface
* Visualization of the final results

To run the 3D grain simulation:

```bash
python grain_3D.py
```

---

## Recommended Directory Structure

```plaintext
│
├─ main.py                 # Main program
├─ params.py               # Simulation parameter definitions
├─ Tracers.py              # Builds tracer layers and motion tracking
├─ grain_3D.py             # 3D grain exposure modeling
├─ display/                # Visualization scripts
│  ├─ draw_impact_dem.py
│  ├─ draw_track.py
│  └─ draw_sample_grains.py
└─ Output/                 # Simulation results
```

---

## Acknowledgment

This code is partially based on the open-source work by **P. O'Brien et al. (2020)**, and has been modified for use in the **Chang’e6** regolith gardening simulation.

If you use this code in your research, please cite both the original **O’Brien et al.** paper and the related **Chang’e6** publication.
