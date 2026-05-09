uv venv --seed
source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
deactivate
uv sync