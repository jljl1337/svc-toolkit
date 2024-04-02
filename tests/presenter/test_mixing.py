from presenter.mixing import MixingPresenter

class MixingWidgetDummy:
    def set_mixer_function(self, function):
        self.function = function

class MixerDummy:
    source_1_path = None
    source_2_path = None
    output_path = None
    source_1_ratio = None
    kwargs = None

    def mix(self, source_1_path, source_2_path, output_path, source_1_ratio, **kwargs):
        MixerDummy.source_1_path = source_1_path
        MixerDummy.source_2_path = source_2_path
        MixerDummy.output_path = output_path
        MixerDummy.source_1_ratio = source_1_ratio
        MixerDummy.kwargs = kwargs

class MixerFactoryDummy:
    def create(self):
        return MixerDummy()

def test_mixing_presenter_constructor():
    # Create a MixingWidget dummy object
    view = MixingWidgetDummy()
    # Create a MixerFactory dummy object
    mixer_factory = MixerFactoryDummy()
    # Create a MixingPresenter object
    presenter = MixingPresenter(view, mixer_factory)

    # Check if the view is set
    assert presenter.view == view
    # Check if the mixer factory is set
    assert presenter.mixer_factory == mixer_factory

    # Check if the mixer function is set
    assert view.function == presenter.mix

def test_mixing_presenter_mix():
    # Create a MixingWidget dummy object
    view = MixingWidgetDummy()
    # Create a MixerFactory dummy object
    mixer_factory = MixerFactoryDummy()
    # Create a MixingPresenter object
    presenter = MixingPresenter(view, mixer_factory)

    # Check if the mixer is created with the correct parameters
    presenter.mix(
        'source_1_path',
        'source_2_path',
        'output_path',
        0.5,
        key1='value1'
    )

    assert MixerDummy.source_1_path == 'source_1_path'
    assert MixerDummy.source_2_path == 'source_2_path'
    assert MixerDummy.output_path == 'output_path'
    assert MixerDummy.source_1_ratio == 0.5
    assert MixerDummy.kwargs == {'key1': 'value1'}