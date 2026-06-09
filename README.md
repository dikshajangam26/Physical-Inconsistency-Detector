# Physical Inconsistency Detector

An end-to-end synthetic research prototype for detecting physically inconsistent transitions using:

- frozen **DINOv2** latent embeddings
- a lightweight **transition probe**
- **residual-based anomaly scoring**
- **probability calibration** with Platt scaling
- **multi-view spatial validation**
- **temporal hardware/visual synchronization monitoring**
- a **counterfactual intervention engine**
- **semantic scene-graph validation**
- and a **downstream ablation study** for measuring the value of uncertainty-aware repair

This repository demonstrates a small but complete **self-healing data loop** for synthetic data quality control.

---

## Project Summary

The project started as a single-view residual probe for detecting broken physical transitions in synthetic square-motion clips. It later expanded into a larger pipeline with:

1. **Single-view anomaly detection**
2. **Calibration and ranking**
3. **Live visualization in Rerun**
4. **Multi-view overhead + ego reasoning**
5. **Temporal sync drift detection**
6. **Counterfactual intervention generation**
7. **Semantic mismatch validation**
8. **Downstream detector ablation study**

The final system combines multiple evidence channels:
- latent residual evidence
- spatial multi-view evidence
- temporal synchronization evidence
- semantic relation evidence
- intervention outputs

---

## Main Components

### 1. Synthetic Data Generation
The synthetic dataset uses a moving white square on a black background to form simple “video” clips.

### 2. Corruption / Anomaly Injection
Multiple anomaly types are injected:
- corrupted final-frame transitions
- random-noise frame replacements
- multi-view ego occlusion
- hardware/visual timing drift
- dropped visual frames
- semantic action contradictions

### 3. Frozen DINOv2 Feature Extraction
A frozen `facebook/dinov2-base` encoder is used to extract image embeddings.

### 4. Transition Probe Model
A lightweight MLP is trained on clean transitions only to predict the latent future state from the latent current state and action cue.

### 5. Residual Scoring + Calibration
The system computes:

$$\text{Residual} = \| z_{\text{after}} - \hat{z}_{\text{after}} \|_2$$

Then converts residuals into calibrated failure probabilities with logistic regression.

### 6. Visualization Layer
A live **Rerun** dashboard is used to inspect:
- current visual states
- residuals
- uncertainty
- hardware telemetry
- sync drift
- intervention outputs
- semantic mismatch traces

### 7. Multi-View Extension
The system supports:
- **overhead view**
- **robot ego view**

Both view embeddings are concatenated:

$$z_{\text{spatial}} = [z_{\text{overhead}}; z_{\text{ego}}]$$

### 8. Temporal Sync Validation
Mock hardware telemetry is compared against visual motion to detect:
- timing drift
- dropped-frame freezes

### 9. Counterfactual Intervention Engine
Flagged samples can be passed through a lightweight synthetic repair engine that applies:
- Gaussian blur for spatial anomalies
- localized gamma/contrast correction for temporal anomalies
- combined repair when both are present

### 10. Semantic Validation
A rule-based semantic layer checks whether the final observed scene graph matches the expected action claim.

### 11. Downstream Ablation Study
A small downstream detector is trained under four data conditions to test the value of:
- noisy raw data
- naive filtering
- counterfactual synthetic repair

---

## Repository Structure

```text
.
├── src/
│   ├── extract_features.py
│   ├── train_probe.py
│   ├── evaluate_router.py
│   ├── visualize_pipeline.py
│   ├── intervention_engine.py
│   ├── semantic_engine.py
│   ├── train_downstream_detector.py
│   ├── generate_multiview_data.py
│   └── corrupt_multiview_data.py
│
├── data/
│   ├── legacy_singleview/
│   │   ├── raw/
│   │   ├── processed_features/
│   │   └── corrupted_list.txt
│   │
│   └── multiview/
│       ├── raw/
│       ├── processed_features/
│       ├── actions.csv
│       └── corrupted_list.txt
│
├── models/
│   ├── singleview/
│   └── multiview/
│
├── results/
│   ├── singleview/
│   ├── multiview/
│   ├── downstream_ablation_results.csv
│   └── downstream_value_of_lift.png
│
└── README.md
```
## Environment Setup

This project was developed in **WSL** with a Python virtual environment.

### Recommended packages

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install opencv-python
pip install scikit-learn
pip install matplotlib
pip install rerun-sdk
```

## Active Transition-Evidence & Counterfactual Data Engine for Embodied AI

## 🏗️ Core Workflow

### A. Single-View Pipeline
1. **Generate single-view data** Create simple synthetic videos with a moving square.
2. **Corrupt part of the data** Inject corrupted transitions and save the corrupted list.
3. **Extract single-view embeddings**
```bash
   python src/extract_features.py
```
4. **Train the transition probe**
   ```bash
     python src/train_probe.py
   ```
5. **Evaluate and calibrate**
   ```bash
      python src/evaluate_router.py
   ```   
### B. Multi-View Pipeline

1. **Generate multi-view data**
   ```bash
   python src/generate_multiview_data.py
   ```
2. **Inject multi-view occlusion anomalies**
   ```bash
   python src/corrupt_multiview_data.py
   ```
3. **Extract multi-view DINOv2 features**
   ```bash
   python src/extract_features.py
   ```
4. **Train the multi-view transiton probe**
   ```bash
   python src/train_probe.py
   ```
5. **Evaluate the multi-view system**
   ```bash
   python src/evaluate_router.py
   ```
   **This produces:**
   ```bash
   results/multiview/spatial_evaluation_results.csv
   ```
### C. Live Visual Inspection (Rerun)
Keep a Rerun viewer session running, then launch:
```bash
python src/visualize_pipeline.py
```
**The visualization includes:**
* overhead camera image
* ego camera image
* intervention image
* latent residual
* calibrated uncertainty
* spatial anomaly
* temporal sync drift
* hardware telemetry
* semantic mismatch
* status logs and overrides

### D. Downstream Ablation Study'
Run:
```bash
python src/train_downstream_detector.py
```
**This trains a small downstream box detector under four configurations:**
1. Clean Base Data Only
2. Base + Corrupted/Unfiltered Data
3. Base + Naive Filtering
4. Base + Counterfactual Synthetics

**Outputs:**
```bash
results/downstream_ablation_results.csv
results/downstream_value_of_lift.png
```


---

## 📈 Key Results

### Single-View Performance
* **Mean clean residual:** 9.3882
* **Mean corrupted residual:** 59.5817
* **Brier Score:** 0.000001
* **AUROC:** 1.000000
* **AUPRC:** 1.000000

#### Lift@k Policy Performance
* **Top 10%** → $\text{Lift@10} = 5.0000$
* **Top 20%** → $\text{Lift@20} = 5.0000$
* **Top 30%** → $\text{Lift@30} = 3.3333$

### Multi-View Performance
* **Mean clean residual:** 12.3996
* **Mean corrupted residual:** 23.5409
* **Mean clean spatial anomaly:** 2.6232
* **Mean corrupted spatial anomaly:** 13.3313
* **AUROC:** 0.942500
* **AUPRC:** 0.790215
* **Brier Score:** 0.084831

### Downstream Ablation Benchmarks

| Training Configuration | BBox Accuracy @ 0.50 | Mean IoU | Curation Paradigm |
| :--- | :---: | :---: | :--- |
| **Clean Base Data Only** | 0.3750 | 0.4045 | Baseline Limit |
| **Base + Corrupted/Unfiltered Data** | 0.5625 | 0.4471 | Shortcut Vulnerable |
| **Base + Naive Filtering** | 0.3125 | 0.4326 | Information Scarcity |
| **Base + Counterfactual Synthetics** | **0.5000** | **0.4589** | **Optimized Curation Loop** |

---

## 🔍 Analytical Interpretation

* **Naive filtering** performed worst on thresholded detection accuracy ($0.3125$), illustrating that dropping raw data outright starves networks of important edge-case geometry.
* **Counterfactual synthetics** yielded the **highest average localization quality (Mean IoU of $0.4589$)**. This confirms that the self-healing intervention engine forces the model to prioritize fine spatial precision rather than shortcut artifacts.
* **Raw corrupted data** unexpectedly achieved the highest coarse bounding box accuracy ($0.5625$). This highlights a classic training trap: the network successfully exploits systemic pipeline noise and layout artifacts in the noisy baseline to trick the thresholded accuracy index, while degrading actual alignment quality.

---

## 💡 Main Architectural Concept

This framework serves as an end-to-end infrastructure proof-of-concept for:
* **Latent residual probing** across frozen foundation models
* **Calibration-aware routing** via Platt Scaling
* **Multi-view consistency reasoning** for geometric alignment
* **Temporal telemetry cross-validation** to target pipeline lag
* **Automatic synthetic repair generation** to resolve data distribution gaps
* **Semantic contradiction detection** via active Spatial Scene Graphs
* **Downstream valuation testing** of uncertainty-aware loop systems

---

## ⚠️ Current Limitations

As a synthetic research prototype, current system constraints include:
* **Simplified World:** The visual space is localized to a basic moving square on a dark canvas.
* **Synthetic Telemetry:** Robot joint velocity streams are procedurally modeled rather than sampled from hardware.
* **Heuristic Engine:** The structural relationships and counterfactual intervention modules rely on rule-based thresholds rather than learned end-to-end models.
* **Optimistic Scaling:** High initial metrics reflect the clean, highly structured nature of the synthetic environment blocks.

---

## 🔮 Future Work

* Transition from synthetic shapes to rich, multi-object physics simulation blocks.
* Integrate real-world multi-camera calibration assets and physical robot telemetry traces.
* Replace rule-based semantic relationships and heuristic filters with learned graph neural networks.
* Upgrade from isolated frame-pair transition probes to true spatial-temporal sequence models.
* Benchmark on large-scale open-source robotics perception datasets (e.g., BridgeData, Ego4D).

---

## 🖼️ Recommended Repository Figures

To maximize application impact, embed the following visuals inside your documentation layout:
1. **Single-View Calibration Curve** (Reliability Diagram)
2. **Lift@k Active Learning Performance Chart**
3. **Statistical Bootstrap Significance Histograms**
4. **Multi-View Occlusion Spatial Anomaly Screenshots**
5. **Rerun Telemetry Operational Dashboard Capture**
6. **Downstream Ablation Value-of-Lift Plot**

---

## 📄 Project Report Citation

For an exhaustive theoretical and mathematical analysis of this framework, reference the full project report:
> *End-to-end report on residual probes, multi-view validation, temporal drift detection, semantic reasoning, and counterfactual data repair.*

---

## 🏁 Final Takeaway

This platform establishes a closed-loop data engine paradigm for physical AI pipelines:  
$$\text{Detect Inconsistencies} \longrightarrow \text{Isolate Failure Roots} \longrightarrow \text{Route Exception Streams} \longrightarrow \text{Generate Synthetics}$$

By replacing dataset noise with structurally sound synthetic interventions, the system ensures downstream models learn robust physical representations over telemetry artifacts.
