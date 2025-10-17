from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="CP4SBI",
    version="1.0.0",
    description="Conformal Prediction for Simulation Based Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=".",
    author="Anonymous",
    author_email=".",
    packages=["tw_CP4SBI"],
    license="MIT",
    keywords=[
    ],
    install_requires=[
        "numpy>=1.25.0",
        "scikit-learn==1.5.1",
        "scipy>=1.12.0",
        "matplotlib==3.9.2",
        "tqdm==4.66.5",
        "torch>=2.5.1",
        "sbi>=0.24.0",
        "numpyro",
        "jax",
        "sbibm"
    ],
    python_requires=">=3.10",
    zip_safe=False,
)
