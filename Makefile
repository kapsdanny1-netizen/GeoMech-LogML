# GeoMech-LogML — developer shortcuts (optional; all commands in README)
.PHONY: app test cli clean

app:            ## launch the Streamlit dashboard
	PYTHONPATH=. streamlit run geomech_logml/app/streamlit_app.py

test:           ## run the full pytest suite
	PYTHONPATH=. python -m pytest

cli:            ## end-to-end CLI run on synthetic data
	PYTHONPATH=. python scripts/train_eval.py --wells 8 --seed 42 --out-dir outputs/run

clean:          ## remove caches and generated outputs
	rm -rf .pytest_cache outputs/** && find . -name __pycache__ -type d -exec rm -rf {} +
