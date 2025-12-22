# Module Setup & Installation

This module provides the δHBV 2.0 (Daily) and δHBV 2.0 MTS (Hourly) models wrapped in a Basic Model Interface (BMI) for use with NOAA-OWP's NextGen framework.

</br>

## Prerequisites

* **Python:** 3.9+
* **PyTorch:** 2.0+ (CUDA for training, CPU/CUDA for inference)
* **NOAA-OWP NextGen:** 0.3+ (Optional if doing standalone BMI forward)

</br>

## Installation

### (1) Standalone Python Installation

Do this if you plan to use the BMI(s) on their own or develop on this module.

1. Clone the repository:

    ```bash
    git clone https://github.com/mhpi/dhbv2.git
    cd dhbv2
    ```

2. Create a conda/venv environment to isolate dependencies:

    Any Python version >=3.9 can be used.

    ```bash
    conda create -n dhbv2 python=3.9
    conda activate dhbv2

    # or

    uv venv --python=3.9 .venv
    source .venv/bin/activate
    ```

3. Install dependencies:

    We recommend [Astral UV](https://docs.astral.sh/uv/) to install packages (available via `pip install uv`), however standard `pip install` will also work.

    ```bash
    uv pip install .

    # or with development dependencies
    uv pip install .[dev]

    # or in editable mode
    uv pip install -e .
    ```

4. Add model weights:

    For parameterization, dhbv2 uses neural networks that require pre-trained weights to perform optimally. These must be downloaded and installed into dhbv2 manually.

    >If AWS CLI is not installed on your system (try `aws --version`) see [AWS instructions](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) for system-specific directions.

    For daily:

    ```bash
    cd dhbv2

    aws s3 cp s3://mhpi-spatial/mhpi-release/models/owp/dhbv_2.zip ./temp/ --no-sign-request

    unzip ./temp/dhbv_2.zip -d ./temp

    mv ./temp/dhbv_2/ ./ngen_resources/data/dhbv2/model/
    rm -r ./temp
    ```

    or install [from your browser](https://mhpi-spatial.s3.us-east-2.amazonaws.com/mhpi-release/models/owp/dhbv_2.zip) (slower) and move contents to `/ngen_resources/data/dhbv2/model/`.

    For hourly:

    ```bash
    cd dhbv2

    aws s3 cp s3://mhpi-spatial/mhpi-release/models/owp/dhbv_2_mts.zip ./temp/ --no-sign-request

    unzip ./temp/dhbv_2_mts.zip -d ./temp

    mv ./temp/dhbv_2_mts/ ./ngen_resources/data/dhbv2_mts/model/
    rm -r ./temp
    ```

    or install [from your browser](https://mhpi-spatial.s3.us-east-2.amazonaws.com/mhpi-release/models/owp/dhbv_2_mts.zip) (slower) and move contents to `/ngen_resources/data/dhbv2_mts/model/`.

### (2) NextGen Integration

To use this module within the [NextGen framework](https://github.com/NOAA-OWP/ngen):

1. Navigate to your NextGen build or source directory:

    We support two distributions. See each repo for its respective installation instructions.

    * [NOAA-OWP/ngen](https://github.com/NOAA-OWP/ngen)
    * [CIROH-UA/ngen](https://github.com/CIROH-UA/ngen) (AWI fork; for NextGen In A Box)

    ```bash
    cd ngen
    ```

2. Add and initialize dhbv2 as a github submodule:

    ```bash
    git submodule add https://github.com/mhpi/dhbv2 ./extern/dhbv2

    git submodule init
    git submodule update --init --recursive
    ```

3. Add dhbv2 data to `ngen/data/`:

    ```bash
    cd ngen

    mv extern/dhbv2/dhbv2/ngen_resources/data data/.
    ```

4. Add model weights:

    For parameterization, dhbv2 uses neural networks that require pre-trained weights to perform optimally. These must be downloaded and installed into dhbv2 manually.

    >If AWS CLI is not installed on your system (try `aws --version`) see [AWS instructions](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) for system-specific directions.

    ```bash
    cd ngen

    aws s3 cp s3://mhpi-spatial/mhpi-release/models/owp/{model_name}.zip . --no-sign-request

    unzip {model_name}.zip -d /temp

    mv /temp/{model_name}/. data/dhbv2/model/
    rm -r /temp
    ```

5. Build ngen in accordance with instructions for your NextGen distribution. This will install necessary dhbv2 dependencies.

</br>

## Directory Structure

After installation, your structure should look like this within NextGen:

```text
ngen/
├── extern/
│   └── dhbv2/
│       └── dhbv2/
│           ├── src/
│           ├── ngen_resources/   # Configs, realizations, model weights
│           └── scripts/          # Standalone test scripts
├── data/
│   ├── dhbv2                 # Configs, realizations, model weights
│   ├── dhbv2_mts             # Configs, realizations, model weights
│   ├── forcing               # Forcings
│   └── spatial               # Hydrofabric data
└── ...
```
