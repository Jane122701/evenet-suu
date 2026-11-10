# EveNet backbone for Suu event tagging and mass regression

This package adapts the actual pretrained EveNet-Full deterministic backbone to
Suu -> chi chi. It does not implement or invoke your thrust/superjet assignment
algorithm, and does not require repeating foundation-model pretraining.

**Status:** the public checkpoint was downloaded and all **504/504 backbone
tensors** loaded with matching names and shapes. Integration tests passed on CPU
using synthetic ROOT events: conversion, a frozen-backbone epoch, a fine-tuning
epoch, validation, checkpoint reload, and inference. No training or performance
evaluation on your physics samples has been performed. The new heads are initially
random; this package is a training implementation, not a trained Suu tagger.

## What the model does

Raw reconstructed objects and event quantities -> checkpoint normalization ->
GlobalEmbedding + PET + ObjectEncoder -> shared 256-dimensional event vector ->
two new MLP heads:

- Event head: one logit, sigmoid gives signal score.
- Mass head: two real numbers estimating log(m_chi / TeV), log(m_suu / TeV).
  Exponentiating and multiplying by 1000 gives masses in GeV.

Both heads use Linear(256,128), GELU, Linear(128,1 or 2). These are our new task
heads, not pretrained mass estimators. Upstream assignment, segmentation, diffusion
and process-classification heads are not instantiated. The original released
normalization buffers are loaded with the backbone. The deterministic backbone
uses time=0 and no input noise. PET random feature dropping is disabled for fine
tuning, consistent with the upstream fine-tuning example. Other dropout is retained.

No generator mass, process label or signal/background label is an input feature.
The input M_all and M_bjets are measurable reconstructed invariant masses, not
truth targets. No chi-jet assignments are provided. Sorting by pT in the converter
only chooses up to 18 objects; it is unrelated to the old two-superjet sorting.

## Install and obtain the released weights

Use a dedicated Python environment; tested with Python 3.12 and PyTorch 2.6 CPU.
Choose your institution's compatible CUDA PyTorch build for GPU training.

```bash
bash setup.sh
python tests/integration.py --upstream upstream --pretrained weights/checkpoints.20M.a4.last.ckpt
```

The setup pins these upstream revisions:

- EveNet-Full: 06c3181d9a0f7df3e30edbb1c355b6fec8b5a3ad
- Core: 2b4702beef0b823338e26713abc1688b03f6043b
- Hugging Face Avencast/EveNet: afba5b47478f0c25c6a5a5755ac30738925d73ad
- Checkpoint SHA256: 3c8b571328112988fd7186953a2f3155f0e22301a9c8d7e24f520881251a5ef2

The loader uses the released `state_dict`, not `ema_state_dict`, records its hash,
and rejects missing, ambiguous or incompatible backbone tensors. The 99 ignored
checkpoint tensors belong to components outside the retained backbone. The full
backbone contains 18,679,555 state elements, including normalization buffers.
Only a state dictionary that PyTorch can safely load with `weights_only=True` is
accepted. There is no automatic fallback to randomly initialized backbone layers.

## Your current ntuple: what still needs to be supplied

The event-tree definition in `clusteringAnalyzerAll_eventlevel.cc` currently has
AK4 four-vectors, DeepJet scores, and run/luminosityBlock/event IDs. Its event tree
does **not** save MET pt or phi, or a complete selected lepton/tau collection.
Therefore the unchanged tree cannot be passed directly to this EveNet adapter.

`prepare_root.py` provides a converter for a **selected all-hadronic sample with
a documented zero-selected-lepton/tau requirement**. It requires actual corrected
MET pt and phi branches and uses binary DeepJet working-point decisions. It never
invents MET or interprets missing leptons as a verified veto. The example manifest
deliberately leaves `selected_zero_leptons_and_taus=false`; set it true only if the
selection has actually been made. This is a converter prerequisite, not a claim
that your existing selection already applies that veto. If you keep events with
leptons/taus, prepare the general object tensors described below instead; do not
silently apply a new veto to your physics analysis.

The 0.2783 b-tag threshold in the example is your existing UL2018 medium DeepJet
choice. Supply the appropriate year/tagger threshold for each production. Do not
feed the continuous DeepJet score into EveNet's categorical btag slot. An extra
continuous-score input is a possible later adaptation, not part of this checkpoint
compatible first implementation.

Selections, jet cleaning, JEC/JER, MET propagation, lepton overlap removal, triggers,
and data-quality requirements must already be applied by the input producer. Only
selected AK4 jets are used; AK8 jets are not added on top of overlapping AK4 jets.
Do not use the old totHT branch automatically: the converter constructs the
specific global sums it needs from retained input objects.

## Data representation

NPZ shards use `schema='evenet-suu-raw-v1'`, with the feature names stored in
`object_features` and `global_features`. Kinematics are **raw GeV**, angles radians,
phi in [-pi,pi]; do not pre-log or standardize them. Normalization happens in model.py.

| Array | Shape | Meaning |
|---|---|---|
| x | N,18,7 | energy, pt, eta, phi, btag, isLepton, charge |
| globals | N,10 | met, met_phi, nLepton, nbJet, nJet, HT, HT_lep, M_all, M_leps, M_bjets |
| mask | N,18 | True for real objects, False for padding |
| event_key | N | Stable globally unique physical-event identifier |
| label | N | 1 signal, 0 background; -1 allowed only for inference |
| mass_gev | N,2 | Signal truth mass parameters [m_chi,m_suu]; background NaN |
| train_weight | N | Nonnegative training weights; default one |

The first converter is jet-only. The generic model can accept lepton/tau objects
through this same schema: btag=0/1 for jets, 2 for tau objects; isLepton=1 for
selected electrons/muons, and use their physical charge. Construct globals and
object selections consistently with the reference pipeline. Raw feature schema
compatibility is not proof of identical detector response or selection.

For the supplied all-hadronic converter, jets are pT ordered and capped at 18,
isLepton/charge are zero, nbJet counts retained tagged jets, nJet counts all selected
jets before truncation, HT sums retained jet pt, M_all sums retained jet four-vectors,
and M_bjets sums retained tagged-jet four-vectors. Lepton globals are zero only under
the verified selection. These definitions follow the public CMS GridStudy converter
in its all-hadronic limit. That repository's global definitions differ from some
older pretraining utility code; this is a documented downstream convention, not a
claim that every upstream converter is identical. The truncation count is reported.
Check the high-multiplicity loss of information before adopting the 18-object cap.

## Convert and fine tune

Edit `configs/manifest.example.json` into your manifest with real files, tree paths,
production namespaces, MET branch names, selection declarations, and many signal
mass points. The example mass values are illustrative. All QCD HT slices and all
other background processes get label=0 in this downstream task.

```bash
python prepare_root.py manifest.json --output prepared
python train.py train \
  --upstream upstream \
  --pretrained weights/checkpoints.20M.a4.last.ckpt \
  --train prepared/train_*.npz \
  --val prepared/val_*.npz \
  --output runs/suu_v1 \
  --epochs 20 --freeze-epochs 2 --batch-size 128 \
  --backbone-lr 1e-5 --head-lr 1e-3 --mass-weight 1
python train.py predict \
  --upstream upstream --checkpoint runs/suu_v1/best.pt \
  --input prepared/test_*.npz --output predictions.npz
```

These are starting hyperparameters, not optimized choices. The first two epochs
train heads with a frozen backbone (dropout off in the backbone); later epochs
fine tune it with a lower learning rate. Validation selects best.pt using the
joint validation loss. This small single-device trainer loads the selected NPZ
shards into RAM. It is not a distributed, streaming production trainer and does
not implement optimizer-state resume. Use a manageable sample for the first run.

For collision-data inference use `label: null` in the conversion manifest, omit
truth masses, and pass all resulting shards to predict. No truth labels are read
by predict. Output is event_key, event_score, and mass_gev[N,2] for **every event**.
Join predictions back to your original event weights and systematic variations by
event identity. Signed MC physics weights for yield templates remain separate
from the nonnegative training weights. Do not silently take absolute weights as
the statistical prescription for a limit fit.

## Loss and background masses

The classification term is nonnegative-weighted binary cross entropy over all MC
events. The mass term is the weighted mean, over signal events only, of the two
SmoothL1 losses between predicted log(m/TeV) and log(truth mass/TeV), beta=1.
Each term is normalized by its own weight sum. Total = classification + lambda*mass.

Signal selection occurs BEFORE logs/loss evaluation, so NaN background targets
never enter the regression calculation. An all-background batch has zero mass
loss. Both heads still execute on every background/data event: their mass outputs
are learned event observables under a signal-trained mapping, not measurements
of real chi or Suu particles in that background event.

## Splits and physics validation

The converter uses SHA256(event_namespace:run:lumi:event) for a stable 70/15/15
train/val/test partition. Use the same namespace and event IDs for systematic
copies of the same event. Distinct MC productions need distinct namespaces, but
overlapping skims of the same production must share them. Repeated generator IDs
within a production require a better upstream globally unique event identifier;
the converter fails rather than silently treating duplicates as independent.
Train/validation duplicate keys are also rejected. Use nominal events for initial
training; do not concatenate systematic copies as independent examples.

For interpolation tests, remove complete mass points from both train and validation
and reserve them for evaluation; the random split alone does not test unseen masses.
Compare pretrained versus scratch/frozen-backbone baselines with identical datasets,
splits and optimization budgets. This package currently trains only the pretrained
route; the benefit of pretraining has not been measured for your signal.

Before a Combine limit, validate signal bias/resolution across the mass grid,
background mass shapes before/after score cuts, control-to-signal-region closure,
JEC/JER and other relevant variations, and spurious-signal behavior. Do not assume
the old factorized score x mass PDF transfers unchanged to these new observables.
A mass output on background is defined by the network; its distribution must be
modeled from suitable MC/control data. Train over a range of masses to avoid a
single-mass regression becoming an approximately constant mass predictor.

## Source provenance and publication

- Backbone: https://github.com/EveNet-HEP/EveNet-Full and https://github.com/EveNet-HEP/Core
- Weights: https://huggingface.co/Avencast/EveNet
- Paper: https://arxiv.org/abs/2601.17126
- CMS example converter: https://github.com/EveNet-HEP/CMSOpenData-GridStudy,
  inspected at 5dfa2a0ab4f81b5ef440db3c6a29c4c88f12c719, process_data.py.
- CMS Open Data terms: https://opendata.cern.ch/docs/about-cms
- CMS publication oversight: https://cms.cern/org/publications-committee
- CMS-member public-data paper policy (internal):
  https://cms-docdb.cern.ch/cgi-bin/DocDB/ShowDocument?docid=14372

DAS is a discovery/access system, not a publication license. For your existing
CMS B2G analysis using internal data/MC, follow CMS approval before external
release. Public CMS Open Data has CC0 reuse terms and requests dataset DOI citations;
CMS/CERN do not endorse resulting independent works. CMS members also have specific
rules for papers using public data; the linked internal policy was not accessible
here, so its current procedural requirements have not been verified. Discuss the
appropriate paper route with your supervisor, B2G conveners and Publications
Committee before public posting. A standalone ML-methods paper and a CMS search
paper may follow different routes. Training solely on simulation does not by itself
make collaboration-restricted samples or derived results public.
