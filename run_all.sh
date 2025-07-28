# Record start time with date and time format
start_time=$(date +%Y-%m-%d_%H-%M)
logfile="data/output/logs/full_log_$start_time.txt"
runtime_start=$(date +%s)

echo -e "\033[34m- - Running gpt.py - -\033[0m"
time PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src/language_model" uv run src/language_model/gpt.py 2>&1 | tee $logfile

# echo -e "\033[34m- - Running export.py - -\033[0m"
# time PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src/language_model" uv run src/language_model/export.py 2>&1 | tee -a $logfile

# echo -e "\033[34m- - Running streamlit_app.py - -\033[0m"
# time PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src/language_model" streamlit run src/language_model/streamlit_app.py 2>&1 | tee -a $logfile

# Record end time and calculate total runtime
end_time=$(date +%s)
total_runtime=$((end_time - runtime_start))
hours=$((total_runtime / 3600))
minutes=$(( (total_runtime % 3600) / 60 ))
seconds=$((total_runtime % 60))
echo -e "\033[32m- - Total runtime: ${hours}h ${minutes}m ${seconds}s - -\033[0m" | tee -a $logfile
