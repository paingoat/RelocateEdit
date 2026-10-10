import sys
import types
import unittest

from relocate_edit.utils.devices import ensure_pkg_resources, unsupported_cuda_arch_message
from relocate_edit.utils.xformers_compat import architecture_listed, xformers_supports_device

import torch


class CudaArchMessageTest(unittest.TestCase):
    def test_blackwell_is_rejected_by_cuda_118_wheels(self):
        message = unsupported_cuda_arch_message(
            "NVIDIA RTX PRO 4500 Blackwell",
            (12, 0),
            ["sm_80", "sm_86", "sm_90"],
            "2.1.2+cu118",
            "11.8",
        )
        self.assertIn("sm_120", message)
        self.assertIn("bash scripts/01_create_env.sh", message)

    def test_matching_arch_is_accepted(self):
        message = unsupported_cuda_arch_message(
            "NVIDIA RTX PRO 4500 Blackwell",
            (12, 0),
            ["sm_90", "sm_120"],
            "2.7.1+cu128",
            "12.8",
        )
        self.assertIsNone(message)


class XformersArchTest(unittest.TestCase):
    def test_hopper_token_does_not_cover_blackwell(self):
        self.assertTrue(architecture_listed(["8.0", "8.6", "9.0"], 8, 6))
        self.assertFalse(architecture_listed(["8.0", "8.6", "9.0"], 12, 0))

    def test_ptx_and_architecture_suffix_match_the_capability(self):
        self.assertTrue(architecture_listed(["8.0+PTX", "12.0"], 12, 0))
        self.assertTrue(architecture_listed(["9.0a"], 9, 0))
        self.assertFalse(architecture_listed(["9.0a"], 12, 0))

    def test_cpu_does_not_need_an_xformers_kernel(self):
        self.assertTrue(xformers_supports_device(torch.device("cpu")))


class PkgResourcesShimTest(unittest.TestCase):
    def test_missing_module_gets_declare_namespace(self):
        saved = sys.modules.get("pkg_resources")
        sys.modules["pkg_resources"] = None
        package = types.ModuleType("example_ns_pkg")
        package.__path__ = ["/tmp/example_ns_pkg"]
        sys.modules["example_ns_pkg"] = package
        try:
            ensure_pkg_resources()
            import pkg_resources

            pkg_resources.declare_namespace("example_ns_pkg")
            self.assertIn("/tmp/example_ns_pkg", list(package.__path__))
        finally:
            if saved is None:
                sys.modules.pop("pkg_resources", None)
            else:
                sys.modules["pkg_resources"] = saved
            sys.modules.pop("example_ns_pkg", None)


if __name__ == "__main__":
    unittest.main()
