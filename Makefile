init:
    pip install -r requirements.txt

test:
    .tests/py.test

.PHONY: init test