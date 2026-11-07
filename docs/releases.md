# Automatic Windows releases

The `Release Windows` GitHub Actions workflow builds and publishes a release on
every push to the default branch (`master` today; `main` is also supported).
It can also be started manually from the Actions tab on the default branch.
Runs on other branches cannot publish. No personal access token or repository
secret is needed: publication uses the workflow's `GITHUB_TOKEN`.

Versions are `v1.2.N`, where `N` is the workflow run number. Each release is tagged
at the exact source commit built by that run and becomes the latest release.
The Windows x64 ZIP contains the app, CUDA and MSVC runtime DLLs, six evaluator
tables, and dependency licenses. A separate `.sha256` file records its checksum.
An NVIDIA driver and compatible GPU are still required for GPU solving.

The workflow builds pinned FLTK 1.4.5 sources and installs CUDA 12.9 on a
Windows Server 2022 runner with Visual Studio 2022. Both evaluator submodules
are checked out at their committed revisions; no local edits or prebuilt
evaluator library are needed. Builds and CPU tests must pass, and the ZIP's
required contents are checked before publication. Hosted runners do not have
a CUDA GPU, so the CPU/GPU comparison and postflop convergence tests must also
be run on a CUDA-capable machine as described in [testing.md](testing.md).

Failed runs publish no release when a build, test, or package check fails.
Fix the failure and push again, or rerun a failed job. Do not rerun a successfully
published job: its version tag already exists. Concurrent releases are serialized.

To build the same package locally, initialize submodules and configure with your
FLTK build and license paths:

```powershell
git submodule update --init --recursive
cmake -S . -B build-release -G "Visual Studio 17 2022" -A x64 -T "cuda=12.9" "-DFLTK_DIR=C:/fltk-1.4.5/build" "-DUTG_FLTK_LICENSE=C:/fltk-1.4.5/COPYING" "-DUTG_RELEASE_VERSION=1.2.0"
cmake --build build-release --config Release --parallel 4
ctest --test-dir build-release -C Release --output-on-failure --timeout 120
cpack --config build-release/CPackConfig.cmake -C Release -B release-assets
./scripts/verify-release.ps1 -Archive release-assets/UnderTheGun-1.2.0-windows-x64.zip
```
