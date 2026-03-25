# Power Market Impacts of Nuclear Energy in Hydropower-Dominated Power Systems

Scripts and data associated with the following study:

> Hjelmeland, M. and Nøland, J. K. (2026). *Power Market Impacts of Nuclear Energy in Hydropower-Dominated Power Systems*. Norwegian University of Science and Technology (NTNU). Posted: 25 Mar 2026. Available at SSRN: https://ssrn.com/abstract=6467238 or http://dx.doi.org/10.2139/ssrn.6467238

---

## Requirements

The EMPS model (via the `pyLTM`) together with the `lpr_sintef_bifrost` package by Lyse Produksjon AS is required to build and run the EMPS model scripts in this repository. All result files and heavier input data files will be available for download from Zenodo.

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
