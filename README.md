# Language model project, Umeå University 2025


In this project I build a language model and this project is a part of the course "Deep Learning - Methods and applications" at Umeå Unversity the year 2025

The beginning of the project was based on [nanogpt-lecture](https://github.com/karpathy/ng-video-lecture) by Andrej Karpathy, which is licensed under MIT license. See [Original Project: nanogpt-lecture](#original-project-nanogpt-lecture) below. Most of the original code have been modified and many additional features has been added.

## Project report

The project report is available in this repo in the file:

[Project report - Deep learning, Methods and applications 2025 - Thomas Selin.pdf](./Project%20report%20-%20Deep%20learning%2C%20Methods%20and%20applications%202025%20-%20Thomas%20Selin.pdf)


## Training data

Pre-training and base training were performed using three datasets, and fine-tuning was conducted with one additional dataset. For detailed information about the datasets and their usage, see the [Project report](#project-report).

## How to run

1. Use [uv](https://github.com/astral-sh/uv) to set up the virtual environment
2. Change venv's python version e.g.: `uv venv --python 3.11.9`. Then activate the venv with `source .venv/bin/activate`
3. Install dependencies: `uv sync`
4. Run the `run_all.sh` file to both create, train, export and serve/inference the model via a web app.

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


### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
Both the original work by Andrej Karpathy, modifications and additions are covered under this license.
