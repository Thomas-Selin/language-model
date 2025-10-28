# Language model project, Umeå University 2025

In this project I've built/trained a language model. This project was a part of the course "Deep Learning - Methods and applications" at Umeå University the year 2025, but it has been updated since then.

The beginning of the project was based on [nanogpt-lecture](https://github.com/karpathy/ng-video-lecture) by Andrej Karpathy, which is licensed under MIT license. See [Original Project: nanogpt-lecture](#original-project-nanogpt-lecture) below. Most of the original code has been modified and many additional features have been added.

### ***Developed by Thomas Selin***

## Project report

[**Project report - Deep learning, Methods and applications 2025 - Thomas Selin.pdf**](./Project%20report%20-%20Deep%20learning%2C%20Methods%20and%20applications%202025%20-%20Thomas%20Selin.pdf)

## Model architecture

The model architecture uses the decoder only transformer architecture. It is of the type **G**enerative **P**re-trained **T**ransformer model. The key components of the architecture include:

1. **Embedding Layer**: Converts input tokens into dense vectors.
2. **Transformer Blocks**: Stacked layers of self-attention and feed-forward networks.
3. **Output Layer**: Projects the final hidden states to the vocabulary size for token prediction.

**NOTE:** In `config.py` you should adjust parameters related to architecture, such as embeddings size and number of attention heads. As per default they are set extremely conservatively to easily enable running the whole training and inference process quickly and on limited hardware. See the following diagram for a high-level overview of one of these architectures.

[Model architecture diagram](./images/model_architecture.png)

## Training data

Pre-training/base-training were performed using three datasets, and fine-tuning was conducted with one additional dataset. For detailed information about the datasets and their usage, see the [Project report](#project-report).

**NOTE:** The files in the `data/input/parquet_files` folder will be deleted after being used for training unless you set AUTO_DELETE_USED_FILES=False in `config.py`.

## Already trained models

The base-trained model and fine-tuned/chat-aligned models created in the project are available on request (369 Mb each).

## How to train and then use the model via the user interface

**Requirements**: Python 3.11 (tested with 3.11.9), [uv package manager](https://github.com/astral-sh/uv)

**GPU**: If a CUDA-capable GPU is available when training and running the model, it will be used. Otherwise, it will fallback to MPS (Apple) acceleration and in last resort CPU.

**Note**: To train a model on your own data, you'll need to replace the example training data for the base-training in the `data/input/parquet_files` folder. You should also replace the example question-answer dataset used for fine-tuning, place it here `data/input/chat-align/question_answer_dataset.parquet`. If you just want to start the UI and use the model files that I can provide on request, you can skip the training step by uncommenting `NO_TRAINING=true` at the top in `run_all.sh` and placing the model files in the `data/output/` folder.

Use [uv](https://github.com/astral-sh/uv) to set up the virtual environment:

1. Set venv's python version and create venv: `uv venv --python 3.11`. Then activate the venv with `source .venv/bin/activate`
2. Install dependencies: `uv sync`
3. Either run `./run_all.sh` file to both create, train, export and serve the model via a web app in which you can inference/call the model. Alternatively, you can skip training and use and already created model, such as the example models, by running `NO_TRAINING=true ./run_all.sh`. 

### Monitoring the training process

Run `tensorboard --logdir=data/output/tensorboard_logs` to start the TensorBoard UI for monitoring training.

## Adjusting training parameters during training process

The training process supports dynamic runtime overrides, allowing you to adjust key parameters without restarting or modifying the main configuration files. This is useful for tuning hyperparameters, batch sizes, learning rates, or other settings while training is in progress.

- **Override file location:** Place your overrides in the `RUNTIME_OVERRIDES.json` file at the project root.
- **Supported parameters:** Any parameter defined in `src/language_model/config.py` can be overridden. Common examples include `learning_rate`, `batch_size`, and `max_iters`.
- **How it works:** During training, the system periodically checks for updates in `RUNTIME_OVERRIDES.json`. If changes are detected, the new values are applied immediately to the running process.
- **Example override file:**

```json
{
  "learning_rate": 0.0005,
  "batch_size": 32,
  "early_stopping_patience": 10000,
  "log_level": "DEBUG"
}
```

## Stopping base training and continue with fine-tuning

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

The models will be saved in pytorch (`.pt`) format in the `data/output/` directory.
The base trained model will be called `best_model.pt` and the chat aligned model will be called `chat_aligned_model.pt`.

## Running tests
To run the tests, use the command:
```bash
PYTHONPATH=src pytest
```

## Original Project: nanogpt-lecture
A project building a GPT which was created in the [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) video lecture series. The code for that project can be found here: [nanogpt-lecture](https://github.com/karpathy/ng-video-lecture) 


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
Both the original work by Andrej Karpathy, modifications and additions are covered under this license.
