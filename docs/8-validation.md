# Runtime Validation

This module includes benchmark runoff simulations for daily and hourly δHBV2.0 models which can be used as a point of comparision to ensure there are no errors with your runtime or module setup.

</br>

## (1) Daily Model

> [Will be added at a later time.]

### (2) MTS (Hourly) Model

The script `./tests/test_mts_bmi.py` can be used to validate your MTS runtime for the catchment `cat-2453` with simulation output from `2009-01-01 00:00:00` to `2010-12-30 23:00:00`.

To validate,

1. Run the BMI standalone example

    ```python
    python scripts/forward_mts_cat-2453.py
    ```

    This will provide an runoff output `./output/dhbv_2_mts_cat-2453_runoff.npy`

2. Run the test

    ```python
    python tests/test_mts_bmi.py
    ```

    This test is designed to accept deviations from the benchmark up to 1e-6, beyond which some error is possible between computers. **If there are errors larger than 1e5** when using the example scripts please submit an [issue](https://github.com/mhpi/dhbv2/issues).

    This test passing confirms the module can replicate benchmark performance for your applications.
