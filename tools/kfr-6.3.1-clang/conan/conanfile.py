"""
Template recipe for an FFT library built with Clang on top of KFR 6.3.1 and consumed by GCC 10 projects.

Build:   KFR_ROOT=<build-clang.sh install prefix> conan create . -pr:h profiles/clang-gcc10rt
Consume: conan install --requires myfft/1.0 -pr:h profiles/gcc10
"""
import os
from conan import ConanFile
from conan.tools.files import copy
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.scm import Version


class MyFFT(ConanFile):
    name = "myfft"
    version = "1.0"
    settings = "os", "arch", "compiler", "build_type"

    def export_sources(self):
        copy(self, "CMakeLists.txt", self.recipe_folder, self.export_sources_folder)
        copy(self, "fft_wrapper.*", os.path.join(self.recipe_folder, "..", "abi_demo"), self.export_sources_folder)

    # Clang versions the binaries are built with (against the GCC 10 libstdc++ runtime)
    clang_versions = ["18", "15", "12"]

    def compatibility(self):
        # The library exposes a C interface and is built with Clang against the GCC 10 libstdc++
        # runtime, so the Clang binaries can be used by consumers building with GCC 10 or newer.
        if self.settings.compiler == "gcc" and Version(self.settings.compiler.version) >= "10":
            # changing the compiler resets its sub-settings, so carry them over explicitly
            keep = [(s, self.settings.get_safe(s)) for s in ("compiler.libcxx", "compiler.cppstd")]
            return [{"settings": [("compiler", "clang"), ("compiler.version", v)] + keep} for v in self.clang_versions]

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.cache_variables["KFR_ROOT"] = os.environ["KFR_ROOT"]  # prebuilt KFR (build-clang.sh output)
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        CMake(self).install()

    def package_info(self):
        self.cpp_info.libs = ["myfft", "kfr_dft_neon64", "kfr_dsp_neon64"]
        self.cpp_info.system_libs = ["m"]
