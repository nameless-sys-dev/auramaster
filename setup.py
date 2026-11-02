import os
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py


class CustomBuildPy(build_py):
    """Custom build hook that compiles the stripped FFmpeg if not already present."""

    def run(self):
        project_root = os.path.dirname(os.path.abspath(__file__))
        ffmpeg_bin = os.path.join(project_root, "auramaster", "bin", "ffmpeg")
        build_script = os.path.join(project_root, "scripts", "build_custom_ffmpeg.sh")

        if not os.path.exists(ffmpeg_bin) and os.path.exists(build_script):
            print("Building embedded audio-only FFmpeg binary...")
            subprocess.check_call(["bash", build_script])

        super().run()


setup(
    name="auramaster",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "auramaster": ["bin/ffmpeg", "bin/ffprobe"],
    },
    cmdclass={
        "build_py": CustomBuildPy,
    },
)