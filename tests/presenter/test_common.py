from vc_toolkit.presenter.common import get_available_device

def test_get_available_device():
    devices = get_available_device()
    assert len(devices) > 0
    assert devices[-1] == ('CPU', 'cpu')