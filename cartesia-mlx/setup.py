import io
import os

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev
import sys
from distutils.util import convert_path
from shutil import rmtree

from setuptools import Command, find_packages, setup

PACKAGE_DIR = "cartesia_mlx"
main_ns = {}
ver_path = convert_path(os.path.join(PACKAGE_DIR, "version.py"))
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)


# Package metadata.
NAME = "cartesia-mlx"
DESCRIPTION = "The official Cartesia MLX library."
URL = "https://github.com/cartesia-ai/edge"
EMAIL = "support@cartesia.ai"
AUTHOR = "Cartesia, Inc."
REQUIRES_PYTHON = ">=3.9.0"
VERSION = main_ns["__version__"]
REQUIRES = [
    "mlx==0.22.1",
    "nanobind==2.4.0",
    "transformers>=4.40.0",
    "datasets",
    "torch",
    "numpy<2",
    "requests",
    "huggingface_hub"
]


# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = [("skip-upload", "u", "skip git tagging and pypi upload")]
    boolean_options = ["skip-upload"]

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        self.skip_upload = False

    def finalize_options(self):
        self.skip_upload = bool(self.skip_upload)

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
            rmtree(os.path.join(here, "build"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        if self.skip_upload:
            self.status("Skipping git tagging and pypi upload")
            sys.exit()

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(
        exclude=[
            "scratch",
            "tests",
            "*.tests",
            "*.tests.*",
            "tests.*",
        ]
    ),
    install_requires=REQUIRES,
    include_package_data=True,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    # $ setup.py publish support.
    cmdclass={"upload": UploadCommand},
)
