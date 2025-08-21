#!/usr/bin/env python3
"""
Simple script to stop base training by creating a stop file.
Run this from another terminal while training is running.
"""

import os
import sys

def stop_training():
    stop_file_path = "../data/output/STOP_TRAINING"
    
    # Create the data/output directory if it doesn't exist
    os.makedirs("../data/output", exist_ok=True)
    
    # Create the stop file
    with open(stop_file_path, 'w') as f:
        f.write("Stop training requested\n")
    
    print(f"✅ Stop signal sent! Created file: {stop_file_path}")
    print("The training will stop at the next check and proceed to fine-tuning.")

if __name__ == "__main__":
    stop_training()
