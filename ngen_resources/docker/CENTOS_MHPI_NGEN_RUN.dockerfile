# Build ngen on CentOS/Rocky Linux 8 with MHPI dependencies and Python3.9
ARG NPROC=4
ARG NGEN_REPO=https://github.com/mhpi/ngen.git
ARG NGEN_BRANCH=mdframe_fix


# Stage 1: Get ngen source and initialize submodules
FROM rockylinux:8 AS source
ARG NPROC
ARG NGEN_REPO
ARG NGEN_BRANCH

RUN yum install -y git \
    # 1. ngen
    && git clone --recursive -b ${NGEN_BRANCH} ${NGEN_REPO} /ngen_src \
    # 2. dmod for multiprocessing
    && git clone -b master https://github.com/NOAA-OWP/DMOD.git /dmod_src \
    # 3. t-route for routing
    && git clone --recursive -b master http://github.com/NOAA-OWP/T-Route.git /troute_src

# Clone and init all submodules
WORKDIR /ngen_src
RUN git submodule update --init --recursive --jobs ${NPROC} --depth 1

# NOTE: Patching test to fix swapped x/y values
RUN sed -i 's/"v,y,x\\n1.000000,1,1\\n2.000000,1,2\\n2.000000,2,1\\n4.000000,2,2\\n"/"v,x,y\\n1.000000,1,1\\n2.000000,2,1\\n2.000000,1,2\\n4.000000,2,2\\n"/g' /ngen_src/test/utils/mdframe_csv_Test.cpp


# Stage 2: Get dependencies
FROM rockylinux:8 AS dependencies

# Header files and build tools (use replace rockylinux gcc~8.5 with gcc~11)
RUN yum update -y \
    && yum install -y epel-release dnf-plugins-core \
    && dnf config-manager --set-enabled powertools \
    && yum install -y \
        gcc-toolset-11-gcc \
        gcc-toolset-11-gcc-c++ \
        gcc-toolset-11-binutils \
        gcc-toolset-11-make \
        gcc-toolset-11-gcc-gfortran \
    && yum install -y \
        tar \
        git \
        make \
        cmake \
        python39 \
        python39-devel \
        python39-numpy \
        bzip2 \
        udunits2-devel \
        texinfo \
        sqlite-devel \
        netcdf-devel \
        netcdf-cxx4-devel \
        netcdf-fortran-devel \
    && yum clean all \
    && rm -rf /var/cache/yum

# Boost setup
RUN curl -L -o boost_1_86_0.tar.bz2 https://sourceforge.net/projects/boost/files/boost/1.86.0/boost_1_86_0.tar.bz2/download \
    && tar -xjf boost_1_86_0.tar.bz2 \
    && rm boost_1_86_0.tar.bz2
ENV BOOST_ROOT="/boost_1_86_0"


# Stage 3: Build python wheels
FROM dependencies AS python-build
ARG NPROC
WORKDIR /build-space

RUN yum install -y findutils git

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvbin/uv
ENV PATH="/uvbin:${PATH}"

COPY --from=source /ngen_src /ngen_src
COPY --from=source /dmod_src /dmod_src
# t-route installs editable dependencies, so any relocations will break pathing.
COPY --from=source /troute_src /app/troute

# Create python virtual environment
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install core requirements
RUN uv pip install "numpy<2.0.0" pandas scipy bmipy

# Install submodule requirements (e.g., sloth, dhbv2)
RUN find /ngen_src/extern -maxdepth 2 -mindepth 2 -type d \
    -exec uv pip install {} --python-version 3.9 \;

# Install dmod
WORKDIR /dmod_src
RUN uv pip install -r requirements.txt \
    && ./scripts/update_package.sh python/lib/communication \
    && ./scripts/update_package.sh python/lib/modeldata \
    && ./scripts/update_package.sh python/services/subsetservice

# Install t-route
RUN uv pip install \
    xarray netcdf4 joblib toolz Cython pyyaml geopandas pyarrow deprecated wheel
WORKDIR /app/troute
RUN uv pip install -r requirements.txt
RUN export NETCDF="/usr/lib64/gfortran/modules" \
    && scl enable gcc-toolset-11 -- ./compiler.sh


# Stage 4: Build ngen
FROM dependencies AS build
ARG NPROC

COPY --from=python-build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV Python3_ROOT_DIR="/opt/venv"

SHELL ["/usr/bin/scl", "enable", "gcc-toolset-11", "--", "/bin/bash", "-c"]

COPY --from=source /ngen_src /ngen
WORKDIR /ngen

RUN cmake \
    -S . \
    -B build \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    \
    -DNetCDF_ROOT=/usr \
    -DBOOST_ROOT=/boost_1_86_0 \
    -DBoost_NO_BOOST_CMAKE:BOOL=TRUE \
    -DBoost_NO_SYSTEM_PATHS=ON \
    \
    -DNGEN_IS_MAIN_PROJECT=ON \
    -DNGEN_WITH_NETCDF:BOOL=ON \
    -DNGEN_WITH_SQLITE:BOOL=ON \
    -DNGEN_WITH_UDUNITS:BOOL=ON \
    -DNGEN_WITH_MPI:BOOL=OFF \
    -DNGEN_WITH_BMI_FORTRAN:BOOL=ON \
    -DNGEN_WITH_BMI_C:BOOL=ON \
    -DNGEN_WITH_PYTHON:BOOL=ON \
    -DNGEN_WITH_ROUTING:BOOL=ON \
    -DNGEN_WITH_TESTS:BOOL=ON \
    -DNGEN_QUIET:BOOL=ON \
    -DUDUNITS_QUIET:BOOL=ON \
    \
    -DNGEN_WITH_EXTERN_SLOTH:BOOL=ON \
    -DNGEN_WITH_EXTERN_TOPMODEL:BOOL=ON \
    -DNGEN_WITH_EXTERN_CFE:BOOL=ON \
    -DNGEN_WITH_EXTERN_PET:BOOL=ON \
    -DNGEN_WITH_EXTERN_NOAH_OWP_MODULAR:BOOL=ON

RUN cmake \
    --build build \
    --target all \
    -- \
    -j ${NPROC}


# --- Stage 5: Runtime (make image ~3x smaller) ---
FROM rockylinux:8

# Reinstall full runtime dependencies
RUN yum update -y \
    && yum install -y epel-release dnf-plugins-core \
    && dnf config-manager --set-enabled powertools \
    && yum install -y \
        gcc-toolset-11-libstdc++-devel \
        python39 \
        udunits2 \
        sqlite \
        netcdf \
        netcdf-cxx4 \
        netcdf-fortran \
        libgfortran \
        libaec \
        findutils \
    && yum clean all

WORKDIR /app
COPY --from=python-build /opt/venv /opt/venv
COPY --from=python-build /app/troute ./troute
COPY --from=build /ngen/build/ngen .
COPY --from=build /ngen/build/extern ./extern
COPY --from=build /ngen/data /app/data
COPY --from=build /ngen/build/test ./test
COPY --from=build /ngen/test/data ./test/data

# Make sure test-specific libs are indexed
RUN find /app/extern -name "*.so" -exec dirname {} + | sort -u > /etc/ld.so.conf.d/ngen.conf && \
    ldconfig
RUN find /app/test -name "*.so" -printf '%h\n' | sort -u >> /etc/ld.so.conf.d/ngen.conf && \
    ldconfig

ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/lib64:/app/extern:/opt/rh/gcc-toolset-11/root/usr/lib64"
ENV UDUNITS2_XML_PATH="/usr/share/udunits/udunits2.xml"

# Set up python environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Pathing for C++ interpreter
ENV PYTHONHOME="/usr"
ENV PYTHONPATH="/app/troute/src/troute-network:/app/troute/src/troute-routing:/app/extern:/opt/venv/lib/python3.9/site-packages:/opt/venv/lib64/python3.9/site-packages"

RUN ln -s /opt/venv /app/venv

CMD ["./ngen", "data/catchment_data.geojson", "", "data/nexus_data.geojson", "", "data/example_realization_config.json"]
