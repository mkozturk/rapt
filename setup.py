import setuptools
with open("README.md","r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="rapt",
	version="0.0.1",
	author="Kaan Öztürk",
	author_email="mkaanozturk@gmail.com",
	description="Rice Adaptive Particle Tracer",
	long_description=long_description,
	long_description_content_type="text/markdown",
	packages=setuptools.find_packages(),
    install_requires=["numpy","scipy"],
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
	python_requires=">=3.6",
)
