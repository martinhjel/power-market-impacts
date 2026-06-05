# Power Market Impacts of Nuclear Energy in Hydropower-Dominated Power Systems

Scripts and data associated with the following study:

> Hjelmeland, M. and Nøland, J. K. (2026). *Power Market Impacts of Nuclear Energy in Hydropower-Dominated Power Systems*. Norwegian University of Science and Technology (NTNU). Posted: 25 Mar 2026. Available at SSRN: https://ssrn.com/abstract=6467238 or http://dx.doi.org/10.2139/ssrn.6467238

---

## Requirements

Install the public Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

The EMPS model stack (`pyLTM`/`lpr_sintef_bifrost`) is required only to build
datasets, run scenarios, or extract fresh LTM results. These packages may
require separate access/licensing. The paper figure and table scripts read the
compact processed result files in `ltm_processed/` and do not need raw LTM
objects once those files have been created or downloaded.

---

## Case studies

Three main technology deployment cases are studied: offshore wind (**OW**), nuclear (**N**), and combined offshore wind + nuclear (**OWN**). See `calculate_capacity.py` for capacity calculations.

### Load scenarios

| | Offshore Wind | Nuclear | OW + Nuclear |
|---|---|---|---|
| **Linear Load Profile Scaling (LLPS)** | ✓ | ✓ | ✓ |
| **Baseload Addition (BA)** | ✓ | ✓ | ✓ |

- **Linear Load Profile Scaling (LLPS):** Scale the load profile to match required new generation.
- **Baseload Addition (BA):** Add baseload equivalent to new generation.

---

## Repository Layout

- `dataset_builder.py`, `dataset_adjuster.py`, `dataset_runner.py`, and
  `scenario_runner.py` build and run the EMPS scenarios used in the study.
- `nuclear_modeling.py` contains the historic/new nuclear representation used
  by the current scenario construction scripts.
- `scripts/process_ltm_results.py` converts raw LTM outputs to
  `ltm_processed/<model>/<scenario>/processed_data.parquet`.
- `scripts/processed_results.py`, `scripts/merit_order.py`, and
  `scripts/paper/processed_dispatch.py` provide the shared processed-result
  API used by the paper scripts.
- `scripts/paper/` contains the current figure and table generation scripts.
  Older flat scripts under `scripts/` are retained for backwards compatibility,
  but the `scripts/paper/` versions are the canonical current versions.

---

## Reproducing Outputs

Download the input and processed result archives from Zenodo after the record is
published:

```bash
python download_data.py
```

The download helper expects the input data archive as `data.zip` and the
processed result archive as `ltm_processed.tar.gz` by default. Update the
Zenodo record ID in `download_data.py` after upload.

If you have raw LTM outputs and access to the EMPS/LTM stack, create or refresh
the compact processed files with:

```bash
python extract_results.py --model-folder PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load_imp_nuke --workers 4
```

or call the processing script directly:

```bash
python scripts/process_ltm_results.py \
  --model-folder PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load_imp_nuke \
  --workers 4
```

Generate the paper figures and tables from processed results with:

```bash
python scripts/paper/run_all_paper_outputs.py \
  --model-folder PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load_imp_nuke
```

To run only a specific paper output group:

```bash
python scripts/paper/run_all_paper_outputs.py --only revenue surplus
```

The outputs are written to:

```text
visualizations/<model_folder>/paper/
```

---

## Scenario Construction

The scenario runner supports selecting individual scenarios:

```bash
python scenario_runner.py \
  --improve-nuclear-rep \
  --only BASELINE_23TWh_BA BASELINE_23TWh_LLPS
```

The improved nuclear representation writes to model folders with the
`_imp_nuke` suffix, for example:

```text
ltm_output/PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load_imp_nuke/
```

---

## Data sources and parameters

### Capacity factors — offshore wind profiles

| Site | Capacity factor |
|---|---|
| NO2 Sørlige Nordsjø II | 0.5594 |
| NO2 Utsira Nord | 0.5007 |
| NO5 Vestavind D | 0.4578 |

Renewable profiles from the Norwegian Water Resources and Energy Directorate (NVE):
- [Weather datasets for power system models](https://www.nve.no/energi/analyser-og-statistikk/vaerdatasett-for-kraftsystemmodellene/)
- [Norwegian offshore wind sites](https://veiledere.nve.no/havvind/identifisering-av-utredningsomrader-for-havvind/metode-og-vurderinger/beregning-av-kraftproduksjon/)

### Operating costs

Source: [NVE — Costs for power production](https://www.nve.no/energi/analyser-og-statistikk/kostnader-for-kraftproduksjon/)

| Technology | Cost (øre/kWh) | Cost (EUR/MWh) |
|---|---|---|
| Nuclear (operations + fuel) | 31 | 26.4 |
| Offshore wind (50/50 floating/bottom-fixed) | 28.5 | 24.2 |
| Onshore wind | 11 | 9.34 |
| Hydro (konsesjonskraftpris) | 14.7 | 12.5 |

Hydro source: [NVE — Konsesjonskraftpris](https://www.nve.no/konsesjon/konsesjonsbehandling-av-vannkraft/konsesjonskraft-og-konsesjonsavgifter/konsesjonskraftpris/)

---

## Streamlit app

From the project root directory, run:

```bash
python -m streamlit run app/home.py --server.address=0.0.0.0 --server.port=8001 --server.headless=true --server.runOnSave=true
```

---

## References

- [PyLTM API documentation](https://docs.ltm.sintef.energy/ltm-api/guides/timesteps_per_week.html)
