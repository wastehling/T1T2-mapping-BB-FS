# Bloch and Sensitivity Simulations
 
---

This repository reproduces some of the figures from the paper:
**[Simultaneous Volumetric T1 and T2 Mapping With Blood- and Fat-Suppression in Abdominal Aortic Aneurysms](https://doi.org/10.1002/mrm.70297)**
(DOI: [10.1002/mrm.70297](https://doi.org/10.1002/mrm.70297)).

---

## Getting Started

### Prerequisites
- Python 3.8+
- Git

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/wastehling/T1T2-mapping-BB-FS
   cd T1T2-mapping-BB-FS
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Simulations
Execute the main script to generate all figures:
```bash
python run_all.py
```
**Note:** The original simulations for the paper require significant runtime (several hours). For demonstration purposes, some parameters have been adjusted to reduce computation time.

#### Modified Parameters
- `sim_range_delta_phi` in `bloch_simulations/main_fig_bloch_imperfect_spoiling.py`
- `nr_avg` in `sensitivity_analysis/settings.json` (set to 2; the paper used `nr_avg=500`)

### Output
Figures will be displayed during execution and saved in the `figures` directory.

---

## Contact
For questions or feedback, please contact me at w.a.stehling at amsterdamumc.nl

---
