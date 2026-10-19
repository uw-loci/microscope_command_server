"""Why Micro-Manager reports pixel size 0: the presets, what they need, what is set."""

from microscope_command_server.server.handlers.position import describe_pixel_size_presets


class _Setting:
    def __init__(self, device, prop, value):
        self._d, self._p, self._v = device, prop, value

    def get_device_label(self):
        return self._d

    def get_property_name(self):
        return self._p

    def get_property_value(self):
        return self._v


class _Config:
    def __init__(self, settings):
        self._s = settings

    def size(self):
        return len(self._s)

    def get_setting(self, i):
        return self._s[i]


class _Core:
    """The LC-PolScope's MM config: presets keyed on a labelling DObjective."""

    def __init__(self, label):
        self.label = label
        self.presets = {
            "20x 0.4 ORYX": [_Setting("DObjective", "Label", "20x")],
            "4x 0.13 ORYX": [_Setting("DObjective", "Label", "4x")],
        }

    def get_available_pixel_size_configs(self):
        return list(self.presets)

    def get_pixel_size_config_data(self, name):
        return _Config(self.presets[name])

    def get_property(self, device, prop):
        if device != "DObjective":
            raise RuntimeError(f"No device with label '{device}'")
        return self.label


def test_names_the_property_that_does_not_match():
    text = describe_pixel_size_presets(_Core("Objective-F"))
    assert "'20x 0.4 ORYX' needs DObjective.Label=20x (now Objective-F) <- differs" in text
    assert "'4x 0.13 ORYX' needs DObjective.Label=4x (now Objective-F) <- differs" in text


def test_a_missing_device_is_reported_not_raised():
    core = _Core("20x")
    core.presets["odd"] = [_Setting("Ghost", "Label", "x")]
    text = describe_pixel_size_presets(core)
    assert "Ghost.Label=x (now unreadable" in text


def test_no_presets_at_all():
    core = _Core("20x")
    core.presets = {}
    assert "no pixel-size presets at all" in describe_pixel_size_presets(core)
