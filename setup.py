"""Package the research code while retaining the upstream ``diffsynth`` imports."""

from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).parent
requirements = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]
setup(
    name="unified-long-video-in-outpainting",
    version="0.1.0",
    description="Unified long video inpainting and outpainting with overlapping co-denoising",
    long_description=(ROOT / "README.md").read_text(),
    long_description_content_type="text/markdown",
    author="Shuangquan Lyu, Jian Mao, Yue Ma; DiffSynth-Studio contributors",
    url="https://github.com/lvshq/Unified-Long-Video-In-Outpainting",
    packages=find_packages(),
    py_modules=["gen_SSIM_LPIPS", "gen_clip_score", "gen_tLPIPS"],
    install_requires=requirements,
    include_package_data=True,
    package_data={"diffsynth": ["tokenizer_configs/**/**/*.*"]},
    python_requires=">=3.10",
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
    ],
)
