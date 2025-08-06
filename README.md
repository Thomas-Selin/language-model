# Language model


This project builds a language model.

The project is based on [nanogpt-lecture](https://github.com/karpathy/ng-video-lecture) by Andrej Karpathy, which is licensed under MIT license. See [Original Project: nanogpt-lecture](#original-project-nanogpt-lecture) below. Portions of the original code have been modified and many additional features has been added.


## Training data

Pre training/base training was done using a 3 dataset....
Specify which collumns are used in each dataset


Finallly chat alignment fine tuning was done using....



## How to run


1. Use [uv](https://github.com/astral-sh/uv) to set up the virtual environment
2. Change venv's python version e.g.: `uv venv --python 3.11.9`. Then activate the venv with `source .venv/bin/activate`
3. Install dependencies: `uv sync`
4. Run the `run_all.sh` file to both create, train, export and serve/inference the model via a web app.

## Original Project: nanogpt-lecture
A project building a GPT which was created in the [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) video lecture series. The code for the project can be found here: [nanogpt-lecture](https://github.com/karpathy/ng-video-lecture) 


### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
Both the original work by Andrej Karpathy, modifications and additions are covered under this license.
