# Singing Voice Conversion Toolkit

![Tests](https://github.com/jljl1337/svc-toolkit/actions/workflows/tests.yml/badge.svg)
![Deployment](https://github.com/jljl1337/svc-toolkit/actions/workflows/deployments.yml/badge.svg)
[![Codecov](https://codecov.io/gh/jljl1337/svc-toolkit/graph/badge.svg?token=QBM6OLIG00)](https://codecov.io/gh/jljl1337/svc-toolkit)

A self-contained singing voice conversion application using the so-vits-svc architecture, 
with Deep U-Net model for vocal separation feature and easy to use GUI.

## Getting Started

### Installation

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

Note that AMD GPUs are not actively supported, but you can try using the package with the CPU version of PyTorch.

For other installation options, see [Installation](https://jljl1337.github.io/svc-toolkit/installation/).

### Usage

#### Windows

```
svct.exe
```

#### macOS/Linux

```
svct
```

For the detailed usage guide, see [Usage](https://jljl1337.github.io/svc-toolkit/usage/).

## Development

For the detailed development guide, see [Development](https://jljl1337.github.io/svc-toolkit/development/).