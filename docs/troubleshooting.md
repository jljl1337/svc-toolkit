# Troubleshooting

## Reinstalling the package

If the package is not working as expected, try uninstalling and installing the package
again.

If the package is installed using pipx, uninstall the package using this command:

```
pipx uninstall svc-toolkit
```

If the package is installed using virtual environment, uninstall the package by 
deactivating the virtual environment and delete it.

Then, install the package again. For installation instructions, see [here](./installation.md).

## GPU not detected

If the GPU is not detected, make sure that the latest version of the GPU driver
is installed and the correct version of PyTorch is installed. You may uninstall
the package and reinstall it with the correct version of PyTorch.

## Poetry install hangs

If `poetry install` hangs, try running the command with the `-vvv` flag to see what's happening.

```
poetry install -vvv
```

If it stop at something like this:

```bash
[keyring.backend] Loading SecretService
[keyring.backend] Loading Windows
[keyring.backend] Loading chainer
[keyring.backend] Loading libsecret
[keyring.backend] Loading macOS
```

Then you may run this command:

```
poetry config keyring.enabled false
```

[source](https://github.com/python-poetry/poetry/issues/8623)