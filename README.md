# Model-Free Multi-Agent Deep Reinforcement Learning for Hybrid AC/DC Smart-City Microgrids
(WCAIAA 2026)

This repository provides the reference implementation and experimental scaffold
for the paper:

**“Model-Free Deep Reinforcement Learning for Voltage Regulation and Energy Management in Hybrid AC/DC Smart-City Microgrids”**  
Accepted at **WCAIAA 2026**.

The code supports reproducible evaluation of a **fully model-free, multi-agent
reinforcement learning (MARL)** framework for coordinated voltage regulation and
energy management in low-voltage hybrid AC/DC microgrids.

[![DOI](https://zenodo.org/badge/1125365622.svg)](https://doi.org/10.5281/zenodo.18098092)

---

## Overview

- **Control paradigm:** Model-free, decentralized Multi-Agent Reinforcement Learning
- **Agents:**  
  - Three phase-level voltage control agents (A/B/C)  
  - One energy management (EMS) agent
- **Learning algorithm:** Dueling Double Deep Q-Network (D3QN)
- **Baselines:**  
  - Random policy (lower-bound reference)  
  - Supervised ANN baseline trained from MARL trajectories
- **Key performance metrics:**  
  - Voltage deviation (p.u. proxy)  
  - Harmonic distortion proxy  
  - Daily grid energy import (kWh)  
  - Energy cost ($)  
  - CO₂ emissions (kg)

All simulations are conducted over a **realistic 24-hour smart-city operating
profile** with unbalanced three-phase loads, renewable generation, dynamic prices,
and CO₂ intensity signals.

---


## Reproducing the Main Results
### 1. Environment setup
Install the required Python packages (Python ≥ 3.9 recommended):

```bash
pip install -r requirements.txt

License
This project is released under the MIT License, allowing reuse, modification,
and extension with proper attribution.

If you use this code, please cite the WCAIAA 2026 paper associated with this repository."# D3QN_MARL_ACDC_SmartCity_Microgrids" 



