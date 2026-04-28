# GaN PFC Stability Mapper

A comprehensive Python-based simulation framework for analyzing GaN (Gallium Nitride) Power Factor Correction (PFC) circuit stability and performance optimization.

## Overview

This project automates batch simulation of GaN PFC topologies using LTspice, systematically varying gate-drive resistances (Rg_on, Rg_off) and switching parameters to characterize circuit stability, efficiency, and power factor across design space.

## Features

- **Automated LTspice Batch Runner**: Execute hundreds of simulation cases with parametric sweeps
- **Stability Analysis**: Classify designs as stable/unstable based on electrical and thermal metrics
- **Performance Metrics**: Compute efficiency, power factor, input current THD, output ripple
- **Machine Learning Surrogates**: Train Random Forest models to predict performance without simulation
- **Design Optimization**: Rank configurations by multi-objective score (efficiency, PF, ripple, regulation)
- **Comprehensive Reporting**: Generate HTML briefs, Excel dashboards, and visualization plots

## Scripts

- **gan_pfc_batch_runner_ideal_bus.py** - Ideal bus voltage source template (V2 source)
- **gan_pfc_batch_runner_real_load.py** - Real load/output filter template
- **gan_pfc_batch_runner_v2.py** - Alternative batch runner variant
- **gan_pfc_pipline.py** - Pipeline orchestration utility
- **gan_pfc_batch_runner.py** - Legacy baseline runner

## Templates

- **gan_pfc_template_ideal.asc** - LTspice schematic with ideal AC source (for ideal bus studies)
- Additional templates in project directories

## Requirements

- **Python 3.8+**
- **LTspice** (ADI/Analog Devices LTspice XVII or later)
- **Dependencies** (see requirements.txt):
  - pandas, numpy, matplotlib
  - scikit-learn (for ML surrogates)
  - openpyxl (for Excel export)

## Installation

```bash
git clone https://github.com/ShivamVashishtha/GaN-PFC-Stability-Mapper.git
cd GaN-PFC-Stability-Mapper
pip install -r requirements.txt
```

## Configuration

Edit the `USER CONFIGURATION` section in the batch runner scripts:

```python
LTSPICE_EXE = r"C:\path\to\LTspice.exe"
PROJECT_DIR = Path(r"C:\path\to\schematic\directory")
OUTPUT_DIR = Path(r"C:\output\directory")
TEMPLATE_ASC = PROJECT_DIR / "your_template.asc"

# Run modes: "single_known_good", "rg_sweep", "full_sweep"
RUN_MODE = "single_known_good"
```

## Usage

```bash
# Run ideal bus topology sweep
python gan_pfc_batch_runner_ideal_bus.py

# Run real load topology sweep
python gan_pfc_batch_runner_real_load.py
```

### Output Files

- `gan_pfc_innovation_dataset.xlsx` - Full results table
- `top_10_designs.xlsx` - Best-performing configurations
- `ml_surrogate_report.xlsx` - ML model performance metrics
- `*.png` - Stability maps, optimization scores, performance heatmaps
- `bad_physics_diagnostics.xlsx` - Failed/unstable cases for analysis

## Physics Thresholds

Configurable in each script:

```python
MIN_GOOD_ELAPSED_S = 10.0       # Minimum convergence time
MIN_PIN_W = 1000.0              # Minimum input power
PF_TARGET_MIN = 0.90            # Power factor target
VOUT_TARGET_V = 400.0           # Output voltage target
EFF_TARGET_MIN = 90.0           # Efficiency floor
```

## Optimization Score

Multi-objective score combining:
- **Efficiency** (45%) - Higher is better
- **Power Factor** (25%) - Closer to 1.0 is better
- **Peak Current** (15%) - Lower is better (< 75A)
- **Output Ripple** (10%) - Lower is better (< 20V)
- **Regulation** (5%) - Closer to 400V is better

## Machine Learning Surrogates

Random Forest models trained on stable designs to predict:
- Stability classification
- Optimization score
- Power factor
- Efficiency
- Peak current

Use for rapid design space exploration without running full simulations.

## Advanced Usage

### Full Parameter Sweep

```python
RUN_MODE = "full_sweep"
rg_on_values = [0.5, 1, 2, 3, 5, 7.5]
rg_off_values = [0.5, 1, 2, 3, 5, 7.5]
fsw_values = [65_000, 100_000, 150_000, 200_000]
deadtime_values = [20e-9, 50e-9, 100e-9, 150e-9]
```

### Custom Measurements

Add LTspice .measure directives in `INJECT_EXTRA_MEASURES`:

```python
INJECT_EXTRA_MEASURES = True
EXTRA_MEASURE_DIRECTIVES = [
    ".meas tran my_measure FIND V(node) AT=1m"
]
```

## Troubleshooting

**LTspice not found:**
- Verify `LTSPICE_EXE` path exists
- Check ADI LTspice installation location

**Template errors:**
- Ensure `.param Rg_on=X Rg_off=Y` exists in schematic TEXT directive
- Verify template.net exists alongside template.asc

**No convergence:**
- Check simulation time settings
- Verify component values and topology
- Review LTspice .raw and .log files in output directory

## Author

Shivam Vashishtha

## License

Proprietary - Contact author for licensing terms

## References

- [ADI LTspice Documentation](https://www.analog.com/en/design-center/design-tools-and-calculators/ltspice-simulator.html)
- GaN PFC topology research and design optimization
