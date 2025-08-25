# Record start time with date and time format
# start_time=$(date +%Y-%m-%d_%H-%M)
# logfile="data/output/logs/full_log_$start_time.txt"
logfile="data/output/logs/full_log_2025-08-10_09-52.txt"

runtime_start=$(date +%s)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,roundup_power2_divisions:8"

# echo $PYTORCH_CUDA_ALLOC_CONF
# echo -e "\033[34m- - Running gpt.py - -\033[0m"
# time PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,roundup_power2_divisions:8" PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src/language_model" uv run src/language_model/gpt.py 2>&1 | tee -a $logfile

echo -e "\033[34m- - Running export.py - -\033[0m"
time PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src/language_model" uv run src/language_model/export.py 2>&1 | tee -a $logfile

echo -e "\033[34m- - Running streamlit_app.py - -\033[0m"
time PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src/language_model" streamlit run src/language_model/streamlit_app.py 2>&1 | tee -a $logfile

# Record end time and calculate total runtime
end_time=$(date +%s)
total_runtime=$((end_time - runtime_start))
hours=$((total_runtime / 3600))
minutes=$(( (total_runtime % 3600) / 60 ))
seconds=$((total_runtime % 60))
echo -e "\033[32m- - Total runtime: ${hours}h ${minutes}m ${seconds}s - -\033[0m" | tee -a $logfile
