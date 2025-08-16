# Training Control Guide

## How to Stop Base Training

When your training is running and you see this message:
```
============================================================
WAITING FOR NEW FILES OR USER INPUT
============================================================
Options:
1. Add new .parquet files to continue training
2. Create a file named 'STOP_TRAINING' in the data/output/ folder to stop
   Command: touch data/output/STOP_TRAINING
============================================================
```

You have **3 easy ways** to stop base training and continue to fine-tuning:

### Method 1: Use the shell script (Easiest)
Open a new terminal in the project directory and run:
```bash
./stop_training.sh
```

### Method 2: Use the Python script
Open a new terminal in the project directory and run:
```bash
python stop_training.py
```

### Method 3: Manual command
Open a new terminal and run:
```bash
touch data/output/STOP_TRAINING
```

## What happens when you stop training?

1. ✅ The current model state is saved
2. 🔄 Training stops gracefully 
3. ➡️ The program automatically continues to the fine-tuning phase
4. 🧹 The stop file is automatically cleaned up

## Benefits of this approach

- ✅ **Cross-platform**: Works on macOS, Linux, and Windows
- ✅ **Reliable**: No keyboard interrupt issues
- ✅ **Simple**: Just create a file to signal stop
- ✅ **Safe**: Graceful shutdown with model saving
- ✅ **Continue**: Automatically proceeds to fine-tuning

## File locations

- **Stop file**: `data/output/STOP_TRAINING` (created when you want to stop)
- **Best model**: `data/output/best_model.pt` (saved during training)
