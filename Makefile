PYTHON ?= python
UPSTREAM ?= upstream
CHECKPOINT ?= weights/checkpoints.20M.a4.last.ckpt

.PHONY: test integration check
test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v
integration:
	$(PYTHON) tests/integration.py --upstream $(UPSTREAM) --pretrained $(CHECKPOINT)
check:
	$(PYTHON) -m compileall -q model.py train.py prepare_root.py bootstrap.py tests
	bash -n setup.sh
