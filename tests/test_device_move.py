"""Activating a loaded model must not assign a read-only ``device`` property.

AnyDoor is a LightningModule. The first call moves nothing because the network
is not loaded yet. The second call used to do ``model.device = ...`` and crash
with ``AttributeError: can't set attribute 'device'``.
"""

import unittest

import torch
from torch import nn

from relocate_edit.models.base import BaseModelWrapper
from relocate_edit.utils.devices import sync_stored_device


class _Locked(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))

    @property
    def device(self):
        return self.p.device


class _Stored(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        self.device = "cpu"


class _Parent(_Locked):
    def __init__(self):
        super().__init__()
        self.child = _Stored()


class DeviceMoveTest(unittest.TestCase):
    def test_readonly_device_cannot_be_assigned(self):
        model = _Locked()
        with self.assertRaises(AttributeError):
            model.device = torch.device("cpu")

    def test_second_move_skips_readonly_device(self):
        wrapper = BaseModelWrapper("cpu")
        wrapper.model = _Parent()
        wrapper.model.child.device = "meta"
        wrapper.to("cpu")
        wrapper.to("cpu")
        self.assertEqual(wrapper.model.device, torch.device("cpu"))
        self.assertNotIn("device", vars(wrapper.model))
        self.assertEqual(wrapper.model.child.device, torch.device("cpu"))

    def test_sync_leaves_unrelated_attributes(self):
        module = _Stored()
        module.note = "keep"
        sync_stored_device(module, "cpu")
        self.assertEqual(module.note, "keep")
        self.assertEqual(module.device, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
