# GaN PFC Stability Mapper

This project is basically a hands-on design sandbox for GaN PFC work: it sweeps LTspice simulations, ranks what actually behaves well, and turns the results into something you can read, compare, and act on fast.

The goal is simple: reduce guesswork. Instead of tweaking gate-drive values one at a time and hoping for the best, the batch runners map out the design space and surface the combinations that look stable, efficient, and worth a deeper look.

## What it does

- Runs LTspice cases in batches across gate-drive and switching parameters
- Separates promising designs from obviously bad physics early
- Scores each case with a multi-objective ranking
- Generates datasets, plots, and reports you can review quickly
- Trains lightweight surrogate models for faster exploration after the first pass

## Project layout

- `gan_pfc_batch_runner_ideal_bus.py` — ideal bus / V2-source study
- `gan_pfc_batch_runner_real_load.py` — real load and output-focused study
- `gan_pfc_template_ideal.asc` — LTspice template schematic

## Quick start

```bash
git clone https://github.com/ShivamVashishtha/GaN-PFC-Stability-Mapper.git
cd GaN-PFC-Stability-Mapper
pip install -r requirements.txt
```

Then point the scripts at your local LTspice install and template files in the `USER CONFIGURATION` block.

## Run it

```bash
python gan_pfc_batch_runner_ideal_bus.py
python gan_pfc_batch_runner_real_load.py
```

If you want a broader sweep, switch the run mode inside the script:

```python
RUN_MODE = "full_sweep"
```

## What you get back

Each run produces a clean set of artifacts that make review easier:

- `gan_pfc_innovation_dataset.xlsx` and `.csv` for the full results table
- `top_10_designs.xlsx` for the best-ranked cases
- `ml_surrogate_report.xlsx` for model quality and feature summaries
- `bad_physics_diagnostics.xlsx` for the cases that failed the sanity check
- `*.png` plots for stability, score, PF, efficiency, peak current, and ripple

## Tuning knobs

The runners are set up so you can move fast without digging through the whole script. The most important parameters are:

- `LTSPICE_EXE` — your LTspice executable path
- `PROJECT_DIR` — folder containing the schematic and supporting files
- `OUTPUT_DIR` — where reports and plots are written
- `TEMPLATE_ASC` — the LTspice schematic template to patch per case
- `RUN_MODE` — `single_known_good`, `rg_sweep`, or `full_sweep`

## Scoring logic

The score is intentionally opinionated. It favors designs that are:

- Efficient
- Well-behaved from a power-factor standpoint
- Not abusing peak current
- Keeping output ripple under control
- Holding the output near the target voltage

That keeps the ranking useful for engineering judgment, not just raw numbers.

## Surrogate models

Once you have enough stable cases, the scripts can train Random Forest models to estimate:

- Stability
- Score
- Power factor
- Efficiency
- Peak current

That makes it easier to search the design space without rerunning every possible LTspice case.

## Useful defaults

```python
MIN_GOOD_ELAPSED_S = 10.0
MIN_PIN_W = 1000.0
PF_TARGET_MIN = 0.90
VOUT_TARGET_V = 400.0
EFF_TARGET_MIN = 90.0
```

These are conservative by design. They help keep the output focused on cases that look physically plausible instead of noisy edge cases.

## Custom measurements

If you want to extend the analysis, you can add your own LTspice `.measure` directives:

```python
INJECT_EXTRA_MEASURES = True
EXTRA_MEASURE_DIRECTIVES = [
    ".meas tran my_measure FIND V(node) AT=1m"
]
```

## Troubleshooting

- If LTspice is missing, double-check `LTSPICE_EXE`
- If a template patch fails, confirm the `.param` names exist in the schematic text block
- If a run won’t converge, inspect the generated `.log` and `.raw` files first

## Why this repo exists

GaN PFC design gets interesting fast because small parameter changes can make a big difference. This repo is meant to make that exploration less tedious and more visual — a practical tool for narrowing down good candidates before you spend more time on them.

## Author

Shivam Vashishtha

## License

Proprietary. Contact the author for licensing terms.

## Reference

- [ADI LTspice Documentation](https://www.analog.com/en/design-center/design-tools-and-calculators/ltspice-simulator.html)
