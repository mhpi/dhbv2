To use in the NextGen Framework,
1) Copy the content of the dHBV 2.0 repo into `ngen/extern/dhbv2/dhbv2/`.
2) The contents of `ngen_resources/data/` should then be copied into the ngen repo at `ngen/data/`. This contains data for the model, realizations for NextGen, and other config files enabling the dHBV 2.0 package to be run from within NextGen.

In particular, `ngen_resources/data/` contains:
- dHBV 2.0 model, BMI, and routing configuration files in `config/`,
- Pretrained model weights for dHBV 2.0 in `models/`,
- "Realization" configuration files for NextGen in `realization/`,
- CONUS-scale statistics for static catchment attributes, catchment and nexus data GeoJSON files, and a subset (Juniata River Basin) of the NextGen hydrofabric v2.2 in `spatial/`,
- AORC forcing data for NextGen + dHBV 2.0 forward inference on the Juniata River Basin in `forcing/`.
