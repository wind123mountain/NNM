conda create -n nnm python=3.10
conda activate nnm

pip install uv
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
uv sync --active