# EveNet–Suu

Event-level signal/background tagging and direct chi/Suu mass regression using
the released EveNet-Full pretrained backbone and two new task heads.

**Status: software-tested research prototype. No Suu physics training or physics
performance claim yet.** All 504 backbone state tensors match the released
checkpoint. Unit tests and a CPU integration run with synthetic ROOT events pass.

## Start here

- [Model architecture and loss](docs/architecture.md)
- [Testing: what passed and what remains](docs/testing.md)
- [Detailed data contract and implementation notes](docs/implementation.md)
- [Example input manifest](configs/manifest.example.json)
- [Checkpoint compatibility evidence](validation/checkpoint_audit.json)

## Repository layout

| Path | Responsibility |
|---|---|
| `model.py` | Pretrained normalization/backbone, two new heads, joint loss |
| `prepare_root.py` | Selected AK4 ROOT events -> padded raw-feature NPZ shards |
| `train.py` | Validation, fine tuning, best-checkpoint selection and inference |
| `bootstrap.py` | Pinned upstream source and checksum-verified public weights |
| `setup.sh`, `requirements.txt` | Reproducible dependency setup |
| `configs/` | Example sample/branch configuration |
| `tests/test_core.py` | Fast loss, schema, split and conversion tests |
| `tests/integration.py` | Real checkpoint + synthetic ROOT end-to-end test |
| `docs/` | Architecture, inputs, testing, physics-validation limits |
| `validation/` | Recorded test logs and checkpoint audit; not physics results |

The generated `upstream/`, `weights/`, `prepared/`, and `runs/` directories are
ignored by Git. ROOT datasets, NPZ samples and model weights are not committed.

## Install and test

Python 3.12 was tested. Create your preferred virtual environment first.

```bash
git clone https://github.com/Jane122701/evenet-suu.git
cd evenet-suu
bash setup.sh --cpu
make test
make integration
```

For GPU work, install a compatible PyTorch 2.6 CUDA build and run `bash setup.sh`.
`bootstrap.py` can run repeatedly; it checks existing source/checkpoint content
and does not reset modified upstream files. All tests use synthetic inputs.

## Prepare actual samples and train

The current event tree still needs actual MET pt/phi and a decision on lepton/tau
handling. The supplied converter is for an already selected zero-lepton/tau
sample; do not assert that selection unless it has been applied. For events that
retain these objects, use a general object converter following the documented
schema. Missing measurements must not be replaced with fabricated zeros.

```bash
cp configs/manifest.example.json manifest.json
# Edit paths, branch mappings, production IDs, masses and selection declarations.
python prepare_root.py manifest.json --output prepared
python train.py train --upstream upstream \
  --pretrained weights/checkpoints.20M.a4.last.ckpt \
  --train prepared/train_*.npz --val prepared/val_*.npz \
  --output runs/suu_v1 --epochs 20 --freeze-epochs 2
python train.py predict --upstream upstream --checkpoint runs/suu_v1/best.pt \
  --input prepared/test_*.npz --output predictions.npz
```

Predictions contain the event key, one signal score, and `[m_chi,m_suu]` in GeV
for every event, including backgrounds. Neither truth mass is a model input.
The model does not perform the old thrust/superjet assignment.

Upstream code: [EveNet-Full](https://github.com/EveNet-HEP/EveNet-Full),
[Core](https://github.com/EveNet-HEP/Core).
Paper: [arXiv:2601.17126](https://arxiv.org/abs/2601.17126).
Weights: [Avencast/EveNet](https://huggingface.co/Avencast/EveNet).
Upstream dependencies retain their own licenses. This repository is a private
research adaptation; it is not an official CMS release or an approved analysis.
