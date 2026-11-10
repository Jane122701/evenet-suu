# Testing structure and interpretation

The software tests establish that the implementation runs and obeys its contracts.
They do not establish that EveNet improves your Suu sensitivity or mass resolution.

## Fast tests

Run `make test`. The 13 unittest cases require no upstream download or checkpoint.

| Area | Checks |
|---|---|
| Signal-only mass loss | Changing background targets does not change mass loss; background mass-output gradients from that term are zero |
| Empty signal subset | All-background batches give finite classification loss and zero mass loss |
| Invalid targets/weights | Nonpositive signal masses and negative training weights are rejected |
| Loss accounting | Summing batch numerators/denominators reproduces dataset-level terms |
| Converter | A known two-jet event has the expected total invariant mass, counts, HT and padding |
| Input contract | Wrong feature order and continuous btag values are rejected |
| Data-only inference | NPZ data with no label or mass-target arrays can be loaded |
| Event identity | Duplicate keys and train/validation overlap are rejected; hashing is stable |
| Invalid kinematics | A zero-pt selected jet is rejected |

## Full integration test

Run `make integration`. This uses the actual public pretrained checkpoint and
the pinned, unmodified EveNet backbone code. It creates temporary synthetic ROOT
files, each with 40 signal-like or background-like events. The random kinematic
generation is the same for both classes: these samples are deliberately software
fixtures, not a physics discrimination benchmark.

The test executes the actual converter, trains for one frozen-backbone epoch and
one unfrozen epoch, validates, saves/reloads best.pt, and predicts held-out events.
It checks finite positive masses, scores in [0,1], output counts, and nonzero
gradients reaching the PET input embedding. It deliberately removes a backbone
tensor from a checkpoint and verifies that loading fails. It separately converts
40 unlabeled events and runs inference without truth labels.

The checkpoint audit reports 504/504 matching backbone tensors. The public file's
SHA256 is verified by bootstrap.py before use. Integration test records are in
validation/integration_tests.txt. Unit results are in validation/unit_tests.txt.
The initial package's test record is retained as historical provenance, not a
new validation run. No trained synthetic checkpoint is supplied as a physics model.

## Needed before a physics claim

1. Supply a small representative selected signal/background ntuple with the
   required measured inputs. Inspect units, masks, object selection and truncation.
2. Establish the lepton/tau policy. The jet-only converter must not silently alter
   the analysis selection. If those objects are retained, implement their converter.
3. Fine tune over multiple signal mass points. Keep disjoint test events and
   reserve entire mass points to test interpolation.
4. Compare pretrained, frozen-feature and scratch baselines with controlled
   splits and budgets. The current command implements the pretrained route only;
   a scratch-baseline training option is not yet provided.
5. Measure rejection at fixed signal efficiency and mass bias/resolution/tails
   versus both generated masses, jet multiplicity and kinematics.
6. Check QCD and other background mass shapes before/after score selection, then
   control-region closure, uncertainties and spurious-signal tests for the fit.

The present package does not implement a Combine workspace, physics evaluation
plots, a streaming/distributed trainer, optimizer-state resume, mass decorrelation,
or per-event mass uncertainties. It does not prove score/mass independence.
The existing factorized background PDF must be revalidated for the new observables.
