# Record start time
start_time=$(date +%s)

echo -e "\033[34m- - Running gpt.py - -\033[0m"
time PYTHONPATH="$PWD/src/language_model" uv run src/language_model/gpt.py

echo -e "\033[34m- - Running export.py - -\033[0m"
time PYTHONPATH="$PWD/src/language_model" uv run src/language_model/export.py

echo -e "\033[34m- - Running streamlit_app.py - -\033[0m"
time PYTHONPATH="$PWD/src/language_model" streamlit run src/language_model/streamlit_app.py

# Record end time and calculate total runtime
end_time=$(date +%s)
total_runtime=$((end_time - start_time))
hours=$((total_runtime / 3600))
minutes=$(( (total_runtime % 3600) / 60 ))
seconds=$((total_runtime % 60))
echo -e "\033[32m- - Total runtime: ${hours}h ${minutes}m ${seconds}s - -\033[0m"
