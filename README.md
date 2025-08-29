# Cognitive Therapeutics Engine (CTE)

This repository contains a demonstrative implementation of a Cognitive Therapeutics Engine (CTE) for adaptive, personalized therapy optimization. It includes a core simulation and recommendation engine (`cte_system.py`) and a small Flask-based web UI (`app.py`) to run recommendations and in-silico trials from a browser.

This README explains the repository layout, how to run the code, and provides a focused developer guide for `cte_system.py` so you can extend, test, or integrate it.

## Quick start

1. Activate the virtual environment (PowerShell):

```powershell
.venv\Scripts\Activate
```

2. Install dependencies (if not already installed):

```powershell
python -m pip install -r requirements.txt
```

3. Run the web UI:

```powershell
python app.py
```

Open http://127.0.0.1:5000/ to interact with the UI.

You can also run the demo directly:

```powershell
python -c "import cte_system; cte_system.demo_cte_system()"
python -c "import cte_system; cte_system.comprehensive_cte_demo()"
```

## Repository structure

- `cte_system.py` — Core CTE implementation: dataclasses, mechanistic models, digital twin, agents, optimizer, evaluator, governance, demos.
- `app.py` — Small Flask UI to call recommendations and run virtual trials in the browser.
- `templates/` — HTML templates used by `app.py` (`index.html`, `result.html`).
- `requirements.txt` — Minimal dependencies for running the UI and core code.

## High-level architecture

- Data ingestion: `DataIngest` provides a harmonized `PatientBaseline` for the digital twin.
- Digital twin: `DigitalTwin` composes a PK/PD model and a tumor growth model and runs Monte Carlo simulations to propagate parameter uncertainty.
- Agents: `SafetyAgent` checks safety constraints, `EfficacyAgent` scores efficacy.
- Optimizer: `TherapyOptimizer` generates candidate `TherapyPolicy` objects, simulates them via a `DigitalTwin`, applies agents' checks, and returns a `Recommendation`.
- Evaluation: `CTEEvaluator` provides retrospective validation and safety audits.
- Governance & Monitoring: `ModelGovernance`, `SafetyMonitor` provide audit trails and safety checks.
- Virtual trials: `VirtualTrialEngine` runs simple in-silico randomized trials between two `TherapyPolicy` arms.

## `cte_system.py` — file walkthrough (developer guide)

This section explains the primary components in `cte_system.py` so you can understand, modify, and extend the system.

### 1) Data models (dataclasses)
- `PatientBaseline`: canonical patient representation used to initialize a `DigitalTwin`.
- `TherapyAction`: single dosing action with fields (drug, dose, frequency, duration, start_day).
- `TherapyPolicy`: collection of `TherapyAction` plus metadata (policy_id, created_at).
- `SimulationResult`: container for outputs of a digital twin (time_points, biomarker trajectories, survival and AE risk arrays, uncertainty bounds).
- `Recommendation`: final recommended policy and metadata/explanation.

Type hints are used extensively to keep data shapes explicit.

### 2) Data ingestion (`DataIngest`)
- `fetch_patient(patient_id)` returns a `PatientBaseline`. In the demo this returns simulated data; replace with FHIR/EHR fetch and mapping logic when integrating.
- `harmonize(raw_data)` and `validate_data(patient)` are hooks for data cleaning and validation.

### 3) Mechanistic models
- `MechanisticModel`: abstract base for ODE-based models.
- `PKPDModel`: simple compartmental PK/PD with state [depot, plasma, effect].
  - `ode_system(state, t, dose_schedule)` expects a schedule of (time, amount) tuples and returns state derivatives.
- `TumorGrowthModel`: Gompertz-like tumor growth with a resistance fraction; `ode_system(state, t, drug_effect)` returns derivatives for [tumor_volume, resistant_fraction].

Both models are deterministic; parameter injection and sampling are handled by the `DigitalTwin`.

### 4) Digital twin (`DigitalTwin`)
- Constructed with a `PatientBaseline`. Initializes PK/PD and disease model with patient-specific parameter adjustments.
- `_sample_parameters()` samples multiplicative uncertainties for model parameters (for Monte Carlo sampling).
- `_single_simulation(policy, time_points, perturbed_params)` runs a deterministic time-forward simulation for one sample:
  - Builds dose schedule from `TherapyPolicy`.
  - Uses `odeint` to integrate PK/PD, with a fallback to a simple model if integration fails.
  - Integrates disease progression (tumor model) via explicit Euler steps, with bounds checking and error handling.
  - Returns arrays for tumor volume, drug concentration, survival probability, and AE risk.
- `simulate(policy, horizon, n_samples)` runs `n_samples` simulations with different sampled parameters, aggregates summary statistics (mean, std, uncertainty bounds), and returns `SimulationResult`.

Notes on numerical robustness
- Euler integration is used for the disease model to keep the code simple and robust. Replace with `odeint` or a stiff solver if needed.
- The PKPD ODE uses `odeint` and a simple dose-window heuristic; this is intentionally lightweight. For production, use precise dosing events and ODE solvers that support events.

### 5) Agents and optimizer
- `SafetyAgent.check_safety(policy, simulation)` enforces max-dose per drug and checks aggregate AE risk.
- `EfficacyAgent.score_efficacy(simulation)` computes a composite score from integrated survival and tumor burden.
- `TherapyOptimizer.propose(twin, n_candidates)` generates random candidate policies, filters by safety, scores by efficacy, and returns a `Recommendation`. It also contains helper methods to produce defaults and explanations.

Extensibility
- Replace the random policy generator with a constrained optimizer (e.g., `scipy.optimize`, Bayesian optimization, RL agent) to search the policy space in a structured way.

### 6) Evaluation and governance
- `CTEEvaluator` provides `retrospective_validation` and `safety_audit` utilities that run recommendations across cohorts and compute simple metrics (MAE, a rough AUROC proxy, calibration).
- `ModelGovernance` stores model metadata, deployment logs, and can generate an audit report.

### 7) Virtual trials and monitoring
- `VirtualTrialEngine.run_virtual_trial(control_policy, experimental_policy, n_patients)` runs simplified in-silico trials by calling the `CognitiveTherapeuticsEngine` simulator.
- `SafetyMonitor` contains logic to detect dangerous recommendations and basic drug-interaction checks.

### 8) Top-level API class
- `CognitiveTherapeuticsEngine` ties everything together and provides convenience methods:
  - `get_patient_twin(patient_id)`
  - `simulate_policy(patient_id, policy)`
  - `recommend_therapy(patient_id)`
  - `explain_recommendation(recommendation)`

### 9) Demo functions
- `demo_cte_system()` and `comprehensive_cte_demo()` show example usage flows. They use simulated patient data and are useful for smoke tests.

## Dependencies
- Core: `numpy`, `pandas`, `scipy`.
- Web UI: `flask` (used by `app.py`).

Install them via:

```powershell
python -m pip install -r requirements.txt
```

## How to add new features or integrate

- Replace `DataIngest.fetch_patient` with real EHR/FHIR calls and mapping to `PatientBaseline`.
- Add more mechanistic modules (e.g., immune model) by subclassing `MechanisticModel` and plugging them into `DigitalTwin`.
- Improve optimization: swap out the random search in `TherapyOptimizer._generate_candidates` for a constrained optimizer or RL agent.
- Persist experiment results: add a small database (SQLite) or CSV export in `CTEEvaluator`/`VirtualTrialEngine`.

## Troubleshooting

- Import errors for `pandas`/`numpy`/`scipy`: ensure virtualenv is active and `pip install -r requirements.txt` ran successfully.
- ODE integration failures: check solver inputs (dose schedule, time_points). The code contains fallbacks to prevent total failure; inspect logs for detailed exceptions.
- Performance: current demos use pure-Python loops and `odeint`; scale-up will require vectorization, compiled solvers, or batching.

## Tests & verification

- The demo functions are quick smoke tests. Run them to verify the main code path:

```powershell
python -c "import cte_system; cte_system.demo_cte_system()"
```

For CI, add small unit tests that:
- Validate `DataIngest.fetch_patient` returns a `PatientBaseline`.
- Run a tiny `DigitalTwin.simulate` with n_samples=1 and assert array shapes.

## Next steps (suggested)

- Add authentication and persistent storage to the Flask UI.
- Improve candidate search (constrained optimization or RL).
- Replace heuristic dose events with an event-handling ODE solver and exact dosing.
- Add a small test-suite and a pre-commit linting config.

If you want, I can implement any of the suggested next steps. Tell me which one to prioritize (UI auth, persistent storage, better optimizer, or unit tests).
#   c o g n i t i v e - t h e r a p e u t i c s - e n g i n e  
 #   c o g n i t i v e - t h e r a p e u t i c s - e n g i n e  
 