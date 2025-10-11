# Controls wheter to train a new model or only use a already trained model via the UI
# NO_TRAINING=true

# Record start time with date and time format when you want to create a new log file
start_time=$(date +%Y-%m-%d_%H-%M)
logfile="data/output/logs/full_log_$start_time.txt"

# Ensure log directory exists
mkdir -p data/output/logs

# # Instead, if you want to continue using earlier log file
# logfile="data/output/logs/full_log_2025-08-10_09-52.txt"

runtime_start=$(date +%s)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,roundup_power2_divisions:8"

if [ "$NO_TRAINING" != "true" ]; then
    # Check if required training data exists
    if [ -f "data/input/chat-align/train-00000-of-00001.parquet" ] && [ -n "$(find data/input/parquet_files -name '*.parquet' -print -quit 2>/dev/null)" ]; then
        echo $PYTORCH_CUDA_ALLOC_CONF
        echo -e "\033[34m- - Running gpt.py to train model - -\033[0m"
        time PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,roundup_power2_divisions:8" PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src/language_model" uv run src/language_model/gpt.py 2>&1 | tee -a $logfile
    else
        echo -e "\033[31m- - Skipping training: All required data files not found - -\033[0m"
        echo -e "\033[31m    Required: data/input/chat-align/train-00000-of-00001.parquet - -\033[0m"
        echo -e "\033[31m    Required: parquet files in data/input/parquet_files/ - -\033[0m"
    fi
else
    echo -e "\033[33m- - Skipping training and export (NO_TRAINING=true) - -\033[0m"
fi

# Check if required model files exist before running Streamlit app
if [ -f "data/output/chat_aligned_model.pt" ] && [ -f "data/output/best_model.pt" ]; then
    echo -e "\033[34m- - Running streamlit_app.py to serve model and launch web app - -\033[0m"
    time PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src/language_model" streamlit run src/language_model/streamlit_app.py 2>&1 | tee -a $logfile
else
    echo -e "\033[31m- - Cannot run Streamlit app: All required model files not found - -\033[0m"
    echo -e "\033[31m    Required: data/output/chat_aligned_model.pt - -\033[0m"
    echo -e "\033[31m    Required: data/output/best_model.pt - -\033[0m"
fi

# Record end time and calculate total runtime
end_time=$(date +%s)
total_runtime=$((end_time - runtime_start))
hours=$((total_runtime / 3600))
minutes=$(( (total_runtime % 3600) / 60 ))
seconds=$((total_runtime % 60))
echo -e "\033[32m- - Total runtime: ${hours}h ${minutes}m ${seconds}s - -\033[0m" | tee -a $logfile
