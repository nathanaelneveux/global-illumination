"""Check checkout isolation with the real Cargo config and pass-through wrapper.

This uses two tiny, dependency-free packages with identical Cargo identities and
different source contents, both written before either build. No GPU or shared
production artifacts are needed. Run with python3 tools/test_shared_build_cache.py.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "posix", "the workspace wrapper requires a POSIX shell")
class SharedBuildCacheTest(unittest.TestCase):
    def test_same_package_in_two_checkouts_keeps_its_own_executable(self):
        with tempfile.TemporaryDirectory(prefix="gi-cargo-cache-") as temporary:
            base = Path(temporary).resolve()
            checkouts = [base / "checkout a", base / "checkout b"]
            for checkout in checkouts:
                for directory in [".cargo", "tools", "src"]:
                    (checkout / directory).mkdir(parents=True)
                shutil.copy2(ROOT / ".cargo/config.toml", checkout / ".cargo/config.toml")
                shutil.copy2(ROOT / "tools/rustc-workspace.sh", checkout / "tools/rustc-workspace.sh")
                (checkout / ".cargo/local.toml").write_text(
                    "[build]\nbuild-dir = " + json.dumps(str(base / "shared")) + "\n"
                )
                (checkout / "Cargo.toml").write_text(
                    '[package]\nname = "gi_cache_identity_probe"\n'
                    'version = "0.1.0"\nedition = "2024"\n[workspace]\n'
                )
                (checkout / "src/main.rs").write_text(
                    'fn main() { println!("' + checkout.name + '"); }\n'
                )
            environment = dict(os.environ)
            for key in [
                "CARGO_BUILD_BUILD_DIR", "CARGO_BUILD_TARGET_DIR", "CARGO_TARGET_DIR",
                "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER", "RUSTC_WORKSPACE_WRAPPER",
                "CARGO_BUILD_RUSTC_WRAPPER", "RUSTC_WRAPPER",
            ]:
                environment.pop(key, None)
            # Returning to A checks cache reuse too, not merely two fresh builds.
            for checkout in [*checkouts, checkouts[0]]:
                result = subprocess.run(
                    ["cargo", "run", "--release", "--offline", "--quiet"],
                    cwd=checkout, env=environment, capture_output=True, text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), checkout.name, result.stderr)


if __name__ == "__main__":
    unittest.main()
