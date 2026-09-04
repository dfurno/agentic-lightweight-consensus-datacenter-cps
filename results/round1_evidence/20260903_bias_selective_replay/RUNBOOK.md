# Reconstructed selective-replay runbook

These commands are reconstructed from the versioned scripts, manifests, and recorded commits; the original terminal transcript is unavailable. Run from the repository root with `/root/research/iot-agentic-lightweight-consensus/.venv/bin/python` (Python 3.12.3). Available relevant packages were NumPy 2.4.4, pandas 3.0.3, SciPy 1.17.1, Matplotlib 3.10.9, Pydantic 2.13.4, and PyYAML 6.0.3.

1. At commit `4dcae5c93d05ab4e241530cdb896fa92c25ef23b`, build manifests with `python scripts/build_bias_replay_manifests.py --original /root/research/iot-agentic-lightweight-consensus --run-dir runs/round1/20260903_bias_selective_replay`.
2. For each pair alpha_crp in `{0.3,0.5,0.7}` and M_z in `{2.0,4.0}`, run `python scripts/eval_crp.py /root/research/iot-agentic-lightweight-consensus/results runs/round1/20260903_bias_selective_replay/outputs/<configuration_id> <alpha_crp> <M_z> <configuration_id> --manifest runs/round1/20260903_bias_selective_replay/manifests/<configuration_id>.json`.
3. At commit `6db74cf9514e504d09bac5263dae396339e5c5c5`, run `python scripts/recompose_bias_replay.py --original /root/research/iot-agentic-lightweight-consensus --run-dir runs/round1/20260903_bias_selective_replay --destination publish`.
4. At commit `00f15168d59bbd7860d8bca95a9717db73ba1f1b`, run `python scripts/recompose_bias_dependents.py --original /root/research/iot-agentic-lightweight-consensus --manifest runs/round1/20260903_bias_selective_replay/manifests/acrp0.5_mz4.0.json --output runs/round1/20260903_bias_selective_replay/dependents_publish`.

The run used CPU only and made zero LLM calls. Input and output hashes are in the manifests and validation JSON files. Historical commands and the historical environment are not recoverable.
