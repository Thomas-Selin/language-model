echo -e "\033[34m- - Running gpt.py - -\033[0m"
PYTHONPATH="$PWD/src" uv run src/sbab_gpt/gpt.py
echo -e "\033[34m- - Running export.py - -\033[0m"
PYTHONPATH="$PWD/src" uv run src/sbab_gpt/export.py
echo -e "\033[34m- - Running serving.py - -\033[0m"
PYTHONPATH="$PWD/src" uv run src/sbab_gpt/serving.py
