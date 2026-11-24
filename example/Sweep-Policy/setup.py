"""Editable installation for the Sweep Policy Isaac Lab task package."""

from setuptools import find_packages, setup


setup(
    name="sweeping-policy",
    version="0.1.0",
    description="UR5e shelf-sweeping reinforcement-learning tasks for Isaac Lab",
    packages=find_packages(),
    package_data={"sweeping_policy.src": ["environment.yaml"]},
    include_package_data=True,
    python_requires=">=3.10",
    zip_safe=False,
)
