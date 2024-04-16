# Development

## IDE

[Visual Studio Code](https://code.visualstudio.com/) is recommended as the IDE
for developing the package. You can install the recommended extensions by opening
the workspace in the IDE. Useful commands are also provided as tasks in the
`.vscode/tasks.json` file.

Though, all IDEs that support Python development can be used, and those useful
commands can be copied from the `.vscode/tasks.json` file to the IDE of your choice.

## Poetry

[Poetry](https://python-poetry.org/) is used for managing the package dependencies,
virtual environment, building and publishing the package. It is recommended to install
Poetry globally on your system, instead of in the same virtual environment that installs
the package. The recommended version is `1.8.2`, though newer versions should work.

Using your own choice of virtual environment manager is also possible, but the
steps to install for GPU support might be different.

## Development Environment

To set up the development environment, follow these steps:

1. Install [Python](https://www.python.org/downloads/) (3.10 is recommended, but 3.10 - 3.11 should work)

2. Install [Poetry](https://python-poetry.org/docs/#installation)

3. Clone the repository and checkout a non-main branch

4. Select the Python version to use with Poetry. For example, to use Python 3.10:

        poetry env use path/to/python3.10

5. Install the package in editable mode

        poetry install

6. Select the Python interpreter in the IDE to the virtual environment created by Poetry

7. Upgrade the dependencies if you want to develop with NVIDIA GPU

    Activate the virtual environment created by Poetry if it is not activated:

        poetry shell

    Then, run the following command:

        pip install -U torch==2.1.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

    For CUDA version 11.*, you can change the `cu121` to `cu118`. So the command will be:

        pip install -U torch==2.1.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

8. Good to go!

## Testing

To run the tests, run the following command:

    poetry run pytest

If the poetry environment is activated, you can just run `pytest` straight away:

    pytest