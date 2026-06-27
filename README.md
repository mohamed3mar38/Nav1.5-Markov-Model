# NaV1.5-Markov-Model

Simulation code accompanying the manuscript:

**A Corrected Markov Model Reveals a Fundamental Safety–Efficacy Tradeoff in NaV1.5 Sodium Channel Block**

---

## Repository Structure

```
src/
    step1_markov_core.py
    step3_drug_model.py
    step5_ap_waveform.py
    step6_full_coupled.py
    step7_state_dependent.py
    calibrate_mexiletine.py
    run_all_simulations.py

data/
    corrected_data.json
    mexiletine_calibration_results.json
```

---

## Requirements

Python 3.11+

Required packages:

- NumPy
- SciPy
- Matplotlib

Install using

```bash
pip install -r requirements.txt
```

---

## Running

Run the complete simulation using

```bash
python src/run_all_simulations.py
```

---

## Data

The repository contains all numerical data reported in the manuscript.

---

## Reproducibility

All figures and numerical values reported in the manuscript can be regenerated using the provided scripts.

---

## License

MIT License.