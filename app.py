from flask import Flask, render_template, request, redirect, url_for, jsonify
import traceback

# Import core CTE classes from the main module
from cte_system import (
    CognitiveTherapeuticsEngine,
    TherapyAction,
    TherapyPolicy,
    VirtualTrialEngine,
)

app = Flask(__name__)

# Initialize system (kept in memory for quick interactions)
cte = CognitiveTherapeuticsEngine()

# Simple in-memory store for last results
results_store = {
    "last_recommendation": None,
    "last_simulation": None,
    "last_trial": None,
}


@app.route('/')
def index():
    return render_template('index.html', results=results_store)


@app.route('/recommend', methods=['POST'])
def recommend():
    patient_id = request.form.get('patient_id', '').strip() or 'DEMO_001'
    try:
        rec = cte.recommend_therapy(patient_id)
        sim = cte.simulate_policy(patient_id, rec.policy)

        results_store['last_recommendation'] = rec
        results_store['last_simulation'] = sim

        # Convert some fields to JSON-serializable
        result_payload = {
            'recommendation': {
                'patient_id': rec.patient_id,
                'drug': rec.policy.actions[0].drug,
                'dose': rec.policy.actions[0].dose,
                'duration': rec.policy.actions[0].duration,
                'expected_benefit': rec.expected_benefit,
                'safety_score': rec.safety_score,
                'uncertainty': rec.uncertainty,
            },
            'simulation_summary': {
                'initial_tumor': float(sim.biomarker_trajectories['tumor_volume'][0]),
                'final_tumor': float(sim.biomarker_trajectories['tumor_volume'][-1]),
                'survival_6m': float(sim.survival_probability[min(180, len(sim.survival_probability)-1)]),
                'max_ae_risk': float(sim.adverse_event_risk.max()),
            }
        }

        return render_template('result.html', payload=result_payload)

    except Exception as e:
        tb = traceback.format_exc()
        return render_template('result.html', error=str(e), traceback=tb)


@app.route('/virtual_trial', methods=['POST'])
def virtual_trial():
    try:
        n_patients = int(request.form.get('n_patients', '50'))

        # Build control and experimental policy from form inputs
        ctrl_drug = request.form.get('control_drug', 'cisplatin')
        ctrl_dose = float(request.form.get('control_dose', '75'))

        exp_drug = request.form.get('exp_drug', 'carboplatin')
        exp_dose = float(request.form.get('exp_dose', '120'))

        control_policy = TherapyPolicy(
            actions=[TherapyAction(drug=ctrl_drug, dose=ctrl_dose, frequency=1, duration=21, start_day=0)],
            policy_id='control_web',
            created_at=__import__('datetime').datetime.now()
        )

        experimental_policy = TherapyPolicy(
            actions=[TherapyAction(drug=exp_drug, dose=exp_dose, frequency=1, duration=21, start_day=0)],
            policy_id='exp_web',
            created_at=__import__('datetime').datetime.now()
        )

        vte = VirtualTrialEngine(cte)
        results = vte.run_virtual_trial(control_policy=control_policy, experimental_policy=experimental_policy, n_patients=n_patients)

        results_store['last_trial'] = results

        return render_template('result.html', trial=results)

    except Exception as e:
        tb = traceback.format_exc()
        return render_template('result.html', error=str(e), traceback=tb)


if __name__ == '__main__':
    # Run development server
    app.run(host='127.0.0.1', port=5000, debug=True)
