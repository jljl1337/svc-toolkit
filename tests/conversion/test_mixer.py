import os

from conversion.mixer import MixerFactory, Mixer

def test_mixer_factory():
    # Create a MixerFactory object
    mixer_factory = MixerFactory()

    # Check if the mixer factory is created
    mixer = mixer_factory.create()
    assert isinstance(mixer, Mixer)

def test_mixer_mix():
    # Create a Mixer object
    mixer = Mixer()

    current_dir = os.path.dirname(__file__)

    try:
        exception_str = ''

        # Mix two audio files
        mixer.mix(
            os.path.join(current_dir, 'source.wav'),
            os.path.join(current_dir, 'source_8k.wav'),
            '',
            0.5
        )
    except Exception as e:
        exception_str = str(e)

    assert exception_str == 'Sampling rate mismatch'

    try:
        exception_str = ''

        # Mix two audio files
        mixer.mix(
            os.path.join(current_dir, 'source.wav'),
            os.path.join(current_dir, 'source_shorter.wav'),
            '',
            0.5
        )
    except Exception as e:
        exception_str = str(e)

    assert exception_str == 'Shape mismatch'

    try:
        exception_str = ''

        # Mix two audio files
        mixer.mix(
            os.path.join(current_dir, 'source.wav'),
            os.path.join(current_dir, 'source.wav'),
            '',
            -0.5
        )

    except Exception as e:
        exception_str = str(e)

    assert exception_str == 'Ratio must be between 0 and 1'

    # Mix two audio files

    output_path = os.path.join(current_dir, 'output.wav')

    mixer.mix(
        os.path.join(current_dir, 'source.wav'),
        os.path.join(current_dir, 'source.wav'),
        output_path,
        0.5,
        normalize=True
    )

    assert os.path.exists(output_path)

    os.remove(output_path)

    
