# Troubleshooting

## Poetry install hangs

If `poetry install` hangs, try running the command with the `-vvv` flag to see what's happening.

```bash
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

```bash
poetry config keyring.enabled false
```

[source](https://github.com/python-poetry/poetry/issues/8623)