"""Every name in ``cta.__all__`` must resolve through the lazy loader."""

import cta


def test_every_public_name_resolves() -> None:
    for name in cta.__all__:
        assert getattr(cta, name) is not None, name


def test_aspects_register_in_report_order() -> None:
    # Reports list aspects in registration order; keep the historical order.
    # Other tests may register mock aspects, so compare relative order only.
    from cta import plugins

    device_aspects = [
        "latency",
        "kernels",
        "overlap",
        "os_runtime",
        "roofline",
    ]
    registered = [name for name in plugins.ASPECTS if name in device_aspects]
    assert registered == device_aspects
