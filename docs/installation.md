# Installation

## Option 1: pipx

1. Install [Python](https://www.python.org/downloads/) (3.10 is recommended, but 3.10 - 3.12 should work)

2. Install [pipx](https://pipx.pypa.io/stable/installation/)

3. Install the package by running this following terminal command if you only have one Python version installed:

```
pipx install svc-toolkit
```

To install with a specific Python version, use the `--python` flag. For example, to install with Python 3.10:

```
pipx install svc-toolkit --python 3.10
```

<details markdown>
<summary>Using NVIDIA GPU</summary>

To use the package with NVIDIA GPU, you need to upgrade the following dependencies:

```
pipx inject svc-toolkit torch==2.1.1 torchaudio==2.1.1 --pip-args="-U" --index-url https://download.pytorch.org/whl/cu121
```

For CUDA version 11.*, you can change the `cu121` to `cu118`. So the command will be:

```
pipx inject svc-toolkit torch==2.1.1 torchaudio==2.1.1 --pip-args="-U" --index-url https://download.pytorch.org/whl/cu118
```

</details>

For usage, see [here](./usage.md)

## Option 2: Virtual Environment

Any virtual environment manager can be used to create a virtual environment for the package.
Here is an example using `miniconda`:

1. Install [miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/)

2. Create a new environment:

        conda create -n svc-venv python=3.10

3. Activate the environment:

        conda activate svc-venv

4. Install the package:

        pip install svc-toolkit

Note that `svc-venv` is the environment name, you can change it to any name you like.

<details markdown>
<summary>Using NVIDIA GPU</summary>

To use the package with NVIDIA GPU, you need to upgrade the following dependencies:

```
pip install -U torch==2.1.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
```

For CUDA version 11.*, you can change the `cu121` to `cu118`. So the command will be:

```
pip install -U torch==2.1.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```

</details>

For usage, see [here](./usage.md)

## What is the difference between these two options? Which one should I choose?

For those who are familiar with Python, you may have used `virtualenv` or `venv`
to create isolated environments for your Python projects. This is adopted to avoid
the same dependency with different version conflicts between different projects.

[pipx](https://pipx.pypa.io/en/stable/) is just built on top of `venv`, what it
does is to create a virtual environment for each package you install, and install
the package in that virtual environment. It is usually used for installing packages
with entry point(s), like `svct` in this case.

One of the advantages of using `pipx` is that the entry point of the package is
available in your shell, so you can run the package directly from the terminal without
activating the virtual environment.

However, if you are already familiar with using virtual environment, and you want
to manage the environment yourself, you can use your preferred tools to create one
and install the package in that environment.