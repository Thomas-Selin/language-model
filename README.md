# Language model project, Umeå University 2025


In this project I've built/trained a language model. This project was a part of the course "Deep Learning - Methods and applications" at Umeå Unversity the year 2025.

The beginning of the project was based on [nanogpt-lecture](https://github.com/karpathy/ng-video-lecture) by Andrej Karpathy, which is licensed under MIT license. See [Original Project: nanogpt-lecture](#original-project-nanogpt-lecture) below. Most of the original code have been modified and many additional features has been added.

## Project report

The project report is available in this repo in the file:

[Project report - Deep learning, Methods and applications 2025 - Thomas Selin.pdf](./Project%20report%20-%20Deep%20learning%2C%20Methods%20and%20applications%202025%20-%20Thomas%20Selin.pdf)

## Training data

Pre-training/base-training were performed using three datasets, and fine-tuning was conducted with one additional dataset. For detailed information about the datasets and their usage, see the [Project report](#project-report).

## Already trained models

The base-trained model and fine-tuned/chat-aligned models created in the project are available on request (369 Mb each).

## How to train and run the model via the user interface

**Requirements**: Python 3.11 (tested with 3.11.9), [uv](https://github.com/astral-sh/uv)

**GPU**: If a CUDA-capable GPU is available when training and running the model, it will be used. Otherwise, it will fallback to MPS acceleration and in last resort CPU.

**Note**: If you want to train on your own data, you'll need to replace the sample training data for the base-training in the `data/input/parquet_files` folder. You should also replace the question-answer dataset, place it here `data/input/chat-align/train-00000-of-00001.parquet`. If you just want to start the UI and use the model files that I can provide on request, you can skip the training step by uncommenting `NO_TRAINING=true` at the top in `run_all.sh` and placing the model files in the subfolder in `data/output/<subfolder>` set by the  `TRAINING_START_TIME` value in `config.py`.

Use [uv](https://github.com/astral-sh/uv) to set up the virtual environment:

1. Set venv's python version and create venv: `uv venv --python 3.11.9`. Then activate the venv with `source .venv/bin/activate`
2. Install dependencies: `uv sync`
3. Either run `./run_all.sh` file to both create, train, export and serve the model via a web app in which you can inference/call the model. Alternatively, you can run each step by itself by commenting out the others.

### Monitoring the training process

Run `tensorboard --logdir=.` in the `data/output/<your training session timestamp>/tensorboard_logs` to serve a UI for monitoring training

### Adjusting training parameters during training process

In the file `RUNTIME_OVERRIDES.json` you can override the intial parameters specified in `config.py` during training.

### Stopping base training and continue with fine-tuning

Open a new terminal and run:
```bash
touch data/output/STOP_TRAINING
```

#### What happens when you stop training?

1. ✅ The current model state is saved (data/output/)
2. 🔄 Training stops gracefully 
3. ➡️ The program automatically continues to the fine-tuning phase
4. 🧹 The stop file is automatically cleaned up

#### Model files

The model will be saved in pytorch (`.pt`) format and called `chat_aligned_model.pt`.

## Original Project: nanogpt-lecture
A project building a GPT which was created in the [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) video lecture series. The code for the project can be found here: [nanogpt-lecture](https://github.com/karpathy/ng-video-lecture) 


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
Both the original work by Andrej Karpathy, modifications and additions are covered under this license.
