from setuptools import setup, find_packages

setup(
    name="ttom-cogvideo",
    version="0.1.0",
    description="TTOM: Test-Time Optimization and Memorization for CogVideoX",
    author="TTOM Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "openai",
        "ruamel.yaml",
        "imageio",
        "matplotlib",
        "numpy",
        "scipy",
        "scikit-learn",
        "inflect",
        "Pillow",
        "opencv-python",
        "scikit-video",
    ],
)
