#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Build the recorded private Router source inside its original ARM task image.
set -euo pipefail

router_build_source=${1:?source directory}
router_build_output=${2:?owned build directory}
router_build_action=${3:-build}
mkdir -p "$router_build_output"
cd "$router_build_source"
test -f Cargo.lock
sha256sum Cargo.lock > "$router_build_output/Cargo.lock.before.sha256"

export CUDA_VISIBLE_DEVICES=''
export UV_NO_CONFIG=1
unset UV_CONSTRAINT
export CARGO_BUILD_JOBS=1
export CARGO_HOME="$router_build_output/cargo-home"
export CARGO_TARGET_DIR="$router_build_output/target"
export CMAKE_BUILD_PARALLEL_LEVEL=1
export MAKEFLAGS=-j1
export OMP_NUM_THREADS=1

router_build_tools="$router_build_output/tools"
router_build_apt="$router_build_output/apt"
mkdir -p "$router_build_tools" "$router_build_apt/lists/partial" "$router_build_apt/archives/partial"
router_apt_options=(
  -o "Dir::State::lists=$router_build_apt/lists"
  -o "Dir::Cache::archives=$router_build_apt/archives"
  -o "Dir::Cache::pkgcache=$router_build_apt/pkgcache.bin"
  -o "Dir::Cache::srcpkgcache=$router_build_apt/srcpkgcache.bin"
)
if ! test -f "$router_build_output/build-debs.json"; then
  apt-get "${router_apt_options[@]}" update
  apt-get "${router_apt_options[@]}" --download-only --no-install-recommends -y install \
    cmake libssl-dev pkgconf protobuf-compiler libprotobuf-dev
  python - "$router_build_apt/archives" "$router_build_tools" "$router_build_output/build-debs.json" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

archives, tools, receipt = map(Path, sys.argv[1:])
rows = []
for archive in sorted(archives.glob('*.deb')):
    package = subprocess.check_output(['dpkg-deb', '-f', str(archive), 'Package'], text=True).strip()
    version = subprocess.check_output(['dpkg-deb', '-f', str(archive), 'Version'], text=True).strip()
    # Keep the image's compiler and C runtime when staging build tools.
    extract = not package.startswith(('libc6', 'libc-bin', 'libc-dev', 'linux-libc-dev', 'gcc-', 'libgcc-', 'libstdc++'))
    if extract:
        subprocess.run(['dpkg-deb', '-x', str(archive), str(tools)], check=True)
    with archive.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    rows.append({'file': archive.name, 'package': package, 'version': version, 'sha256': digest,
                 'bytes': archive.stat().st_size, 'extracted': extract})
receipt.write_text(json.dumps(rows, indent=2) + '\n')
PY
fi

if ! test -e "$router_build_tools/usr/bin/pkg-config"; then
  ln -s pkgconf "$router_build_tools/usr/bin/pkg-config"
fi
export PATH="$router_build_tools/usr/bin:$PATH"
export LD_LIBRARY_PATH="$router_build_tools/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PKG_CONFIG="$router_build_tools/usr/bin/pkgconf"
export PKG_CONFIG_LIBDIR="$router_build_tools/usr/lib/aarch64-linux-gnu/pkgconfig:$router_build_tools/usr/share/pkgconfig"
export PKG_CONFIG_SYSROOT_DIR="$router_build_tools"
export OPENSSL_INCLUDE_DIR="$router_build_tools/usr/include"
export OPENSSL_LIB_DIR="$router_build_tools/usr/lib/aarch64-linux-gnu"
# Use the image's shared OpenSSL. Its static archive needs extra compression
# symbols and can produce a wheel that builds but fails at Python import.
for router_ssl_library in ssl crypto; do
  router_ssl_runtime="/usr/lib/aarch64-linux-gnu/lib${router_ssl_library}.so.3"
  test -f "$router_ssl_runtime"
  if ! test -e "$OPENSSL_LIB_DIR/lib${router_ssl_library}.so.3"; then
    ln -s "$router_ssl_runtime" "$OPENSSL_LIB_DIR/lib${router_ssl_library}.so.3"
  fi
done
sha256sum /usr/lib/aarch64-linux-gnu/libssl.so.3 /usr/lib/aarch64-linux-gnu/libcrypto.so.3 \
  > "$router_build_output/native-openssl-sha256.txt"
export OPENSSL_STATIC=0
export CPATH="$router_build_tools/usr/include/aarch64-linux-gnu${CPATH:+:$CPATH}"
export PROTOC="$router_build_tools/usr/bin/protoc"
export PROTOC_INCLUDE="$router_build_tools/usr/include"

rustc --version
cargo --version
cmake --version
"$PROTOC" --version
printf '#include <openssl/ssl.h>\n' | cc -I "$OPENSSL_INCLUDE_DIR" -x c -fsyntax-only -
if ! test -f "$router_build_output/build-python-lock.txt"; then
  if ! test -x "$router_build_output/python/bin/python"; then
    uv venv --python /app/.venv/bin/python "$router_build_output/python"
  fi
  uv pip install --python "$router_build_output/python/bin/python" --index-url https://pypi.org/simple \
    'setuptools>=45' wheel 'setuptools-rust>=1.5.2' build
  uv pip freeze --python "$router_build_output/python/bin/python" > "$router_build_output/build-python-lock.txt"
fi
cargo metadata --locked --format-version 1 --no-deps > "$router_build_output/cargo-metadata.json"
sha256sum --check "$router_build_output/Cargo.lock.before.sha256"
if test "$router_build_action" = preflight; then
  printf 'ROUTER_BUILD_PREFLIGHT_COMPLETE\n'
  exit 0
fi
test "$router_build_action" = build
cargo fetch --locked
export CARGO_NET_OFFLINE=true
"$router_build_output/python/bin/python" -m build --wheel --no-isolation --outdir "$router_build_output/wheels"
sha256sum --check "$router_build_output/Cargo.lock.before.sha256"
sha256sum "$router_build_output"/wheels/*.whl > "$router_build_output/wheel-sha256.txt"
"$router_build_output/python/bin/python" - "$router_build_output" <<'PY'
import subprocess
import sys
import zipfile
from pathlib import Path

output = Path(sys.argv[1])
wheels = list((output / 'wheels').glob('*.whl'))
assert len(wheels) == 1
target = output / 'import-check'
target.mkdir(exist_ok=True)
with zipfile.ZipFile(wheels[0]) as archive:
    extension = next(name for name in archive.namelist() if name.endswith('.so'))
    assert '/' not in extension and extension.startswith('vllm_router_rs.')
    (target / extension).write_bytes(archive.read(extension))
subprocess.run([sys.executable, '-c',
                'import sys; sys.path.insert(0, sys.argv[1]); import vllm_router_rs; print(vllm_router_rs.__file__)',
                str(target)], check=True)
PY
printf 'ROUTER_PRIVATE_WHEEL_BUILD_COMPLETE\n'
