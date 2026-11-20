## 2026-09-01 (Phase 6) â€” Codebase Standardization & Repository Cleanup
â€” Codebase Standardization & Repository Cleanup

**Context**: Preparing the project repository for camera-ready submission and reproducibility archival.

**Actions Taken**:
1. **Network Parameter Standardization**:
   - Created `network_params.json` as the domain-standard configuration for vehicular wireless channels (120ms latency, 5% packet loss, 25 Mbps bandwidth, burst jitter).
   - Updated `ota_core.py` with `load_network_params()` and kept `load_bd_params()` as a backward-compatible alias to prevent breaking legacy scripts.
2. **Paper Scope Verification**:
   - Confirmed `Thesis Paper/draft.tex` is 100% free of local regional artifacts and maintains rigorous IEEE vehicular CPS terminology.
3. **Artifact Audit**:
   - Verified that `results/final/` contains all reconstructed publication-ready plots (`learning_curves.png`, `entropy_curves.png`) and master extracted scalars (`extracted_marl_training_history.csv`).
4. **Type-Conditioning Ablation Switch & Web UI Component**:
   - Implemented `type_conditioning: bool = True` in `marl_ota_env.py` and `--type_conditioning` across `train_mappo.py`, `main.py`, and `tools/evaluate_marl.py`.
   - When enabled (`True`), observations include the 4-dim semantic ECU-type one-hot vector (`[1,0,0,0]` etc.).
   - When disabled (`False` - Blind Mode), observations pass `[0,0,0,0]`, preserving tensor shapes and network architecture while blinding the policy to ECU types for controlled ablation studies.
   - Added the **Type Conditioning** checkbox switch to `webui/index.html`, connected through `webui/app.js` and `web_ui.py` for 1-click execution.
   - Blind runs are automatically tracked under `IPPO_Blind_Safety_True` to keep primary benchmark runs pristine.

---
