# Power Market Impacts of Nuclear Energy in Hydropower-Dominated Power Systems

Scripts and data associated with the following study:

> Hjelmeland, M. and Nøland, J. K. (2026). *Power Market Impacts of Nuclear Energy in Hydropower-Dominated Power Systems*. Norwegian University of Science and Technology (NTNU). Posted: 25 Mar 2026. Available at SSRN: https://ssrn.com/abstract=6467238 or http://dx.doi.org/10.2139/ssrn.6467238

---

## Requirements

Install the public Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

The public figure scripts read compact processed result files in
`ltm_processed/`. Raw EMPS/LTM result folders are not required by this
repository.

The scenario-construction scripts document how the study cases were configured,
but running EMPS itself requires external software and licensing that are not
distributed here.

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
  `scenario_runner.py` document and construct the EMPS scenarios used in the
  study.
- `nuclear_modeling.py` contains the historic/new nuclear representation used
  by the current scenario construction scripts.
- `scripts/processed_results.py` and `scripts/paper/processed_dispatch.py`
  provide the shared processed-result API used by the paper scripts.
- `scripts/paper/` contains the current figure generation scripts used for the
  paper figures.

---

## Reproducing Outputs

Download the input and processed result archives from Zenodo:

```bash
python download_data.py --record-id <zenodo_record_id>
```

The download helper expects these Zenodo files:

```text
data.tar.gz
results_processed.tar.gz
results_processed_imp_nuke.tar.gz
```

You can also set `ZENODO_RECORD_ID=<zenodo_record_id>` instead of passing
`--record-id`.

Generate the paper figures from processed results with:

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
`_imp_nuke` suffix in the original EMPS workflow. The public reproduction
archive contains the corresponding processed output folder:

```text
ltm_processed/PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load_imp_nuke/
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
