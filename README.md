# contrastive_3D_human_body_analysis
Contrastive Pretraining with Hybrid Transformers for 3D Body Shape Analysis
# Mesh Health Predict

Hybrid mesh-transformer pipeline:
- Self-supervised contrastive pretraining on unlabeled meshes (normals + curvature + radial).
- Fine-tuning for downstream clinical tasks:
  - Obstetric: C-section classification + newborn head circumference regression (multitask)
  - Geriatric: multi-target regression for function/performance outcomes

## Setup
```bash
pip install -r requirements.txt
