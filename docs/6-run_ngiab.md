# Running in NextGen-In-A-Box (NGIAB)

[NextGen-In-A-Box](https://github.com/CIROH-UA/NGIAB-CloudInfra) allows you to run the model in a containerized environment (Docker) for rapid and foolproof NextGen deployment.

## (1) Setup

Ensure you have NGIAB cloned and Docker running.

## (2) Mounting the Module

To use this custom external module without rebuilding the Docker image, mount the `dhbv2` directory into the container's `extern` volume.

In your `docker-compose.yml` or run command, add a volume mapping:

```yaml
volumes:
  - /local/path/to/dhbv2:/ngen/extern/dhbv2
```

## (3) Configuration Paths

When running inside the container, ensure your paths in the Realization JSONs are relative to the container's file system or the extern directory.

Example init_config path in JSON:

```json
"init_config": "/ngen/extern/dhbv2/ngen_resources/data/dhbv2_mts/config/bmi_cat-2453.yaml"
```

## (4) Running the Model

Execute the standard NGIAB run command. The container will detect the python package in extern/dhbv2 (assuming extern is in the container's PYTHONPATH).

```bash
# Example execution from NGIAB root
./run_ngen.sh /ngen/extern/dhbv2/ngen_resources/data/dhbv2_mts/realizations/realization_cat-2453.json
```
