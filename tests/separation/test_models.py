import torch
import torch.nn as nn

from separation.models import UNet, DeeperUNet, UNetLightning, _down_layer, _up_layer

def test_down_layer():
    down_layer = _down_layer(10, 20)
    
    assert len(down_layer) == 3
    assert down_layer[0].in_channels == 10

def test_up_layer():
    up_layer = _up_layer(10, 20, dropout=True)
    
    assert len(up_layer) == 4
    assert up_layer[0].in_channels == 10

    up_layer = _up_layer(10, 20, last=True)
    
    assert len(up_layer) == 3
    assert isinstance(up_layer[-1], nn.Sigmoid)

def test_unet_lightning_constructor():
    model = UNetLightning(lr=0.1, weight_decay=0.2)

    assert model.lr == 0.1
    assert model.weight_decay == 0.2
    assert isinstance(model.model, UNet)

    model = UNetLightning(lr=0.1, weight_decay=0.2, deeper=True)
    assert isinstance(model.model, DeeperUNet)

def test_unet_lightning_forward():
    model = UNetLightning()

    x = torch.randn(1, 1, 512, 512)
    y = model(x)

    assert y.shape == (1, 1, 512, 512)
    assert y.min() >= 0
    assert y.max() <= 1

    model = UNetLightning(deeper=True)

    x = torch.randn(1, 1, 512, 512)
    y = model(x)

    assert y.shape == (1, 1, 512, 512)
    assert y.min() >= 0
    assert y.max() <= 1

def test_unet_lightning_optimizer():
    model = UNetLightning(optimizer='adam')

    optimizer = model.configure_optimizers()

    assert isinstance(optimizer, torch.optim.Adam)

    model = UNetLightning(optimizer='adamw')

    optimizer = model.configure_optimizers()

    assert isinstance(optimizer, torch.optim.AdamW)

    try:
        exception_str = ''
        model = UNetLightning(optimizer='sgd')

        optimizer = model.configure_optimizers()
    except Exception as e:
        exception_str = str(e)

    assert exception_str.startswith('Invalid optimizer')

def test_unet_lightning_get_loss():
    model = UNetLightning()

    x = torch.randn(1, 1, 512, 512)
    y = torch.randn(1, 1, 512, 512)

    loss = model.get_loss((x, y))

    assert loss >= 0
    assert loss <= 1

def test_unet_lightning_training_step():
    model = UNetLightning()

    x = torch.randn(1, 1, 512, 512)
    y = torch.randn(1, 1, 512, 512)

    loss = model.training_step((x, y), 0)

    assert loss >= 0
    assert loss <= 1

def test_unet_lightning_validation_step():
    model = UNetLightning()

    x = torch.randn(1, 1, 512, 512)
    y = torch.randn(1, 1, 512, 512)

    loss = model.validation_step((x, y), 0)

    assert loss >= 0
    assert loss <= 1

def test_unet_lightning_on_train_epoch_end():
    model = UNetLightning()

    x = torch.randn(1, 1, 512, 512)
    y = torch.randn(1, 1, 512, 512)

    loss = model.training_step((x, y), 0)

    result = {}

    model.log = lambda x, y: result.update({x: y})

    model.on_train_epoch_end()

    assert 'train_loss' in result
    assert result['train_loss'] == loss

def test_unet_lightning_on_validation_epoch_end():
    model = UNetLightning()

    x = torch.randn(1, 1, 512, 512)
    y = torch.randn(1, 1, 512, 512)

    loss = model.validation_step((x, y), 0)

    result = {}

    model.log = lambda x, y: result.update({x: y})

    model.on_validation_epoch_end()

    assert 'val_loss' in result
    assert result['val_loss'] == loss

def test_unet_lightning_predict_step():
    model = UNetLightning()

    x = torch.randn(1, 1, 512, 512)

    y = model.predict_step(x, 0)

    assert y.shape == (1, 1, 512, 512)





    