echo -e "\033[34m- - Running gpt.py - -\033[0m"
PYTHONPATH="$PWD/src/sbab_gpt" uv run src/sbab_gpt/gpt.py
echo -e "\033[34m- - Running export.py - -\033[0m"
PYTHONPATH="$PWD/src/sbab_gpt" uv run src/sbab_gpt/export.py
echo -e "\033[34m- - Running streamlit_app.py - -\033[0m"
PYTHONPATH="$PWD/src/sbab_gpt" streamlit run src/sbab_gpt/streamlit_app.py
