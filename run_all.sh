echo -e "\033[34m- - Running gpt.py - -\033[0m"
PYTHONPATH="$PWD/src/language_model" uv run src/language_model/gpt.py
echo -e "\033[34m- - Running export.py - -\033[0m"
PYTHONPATH="$PWD/src/language_model" uv run src/language_model/export.py
echo -e "\033[34m- - Running streamlit_app.py - -\033[0m"
PYTHONPATH="$PWD/src/language_model" streamlit run src/language_model/streamlit_app.py
