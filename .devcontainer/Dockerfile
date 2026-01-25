ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-trixie

# Install Java and other dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-21-jdk \
    git \
    wget \
    procps \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Set environment variables
RUN export JAVA_HOME="$(dirname $(dirname $(readlink -f $(which java))))"

# Arguments for Spark setup
ARG SPARK_VERSION=3.5.7
ARG HADOOP_VERSION=3.4.2

# Set Spark environment variables
ENV SPARK_VERSION=${SPARK_VERSION}
ENV HADOOP_MAJOR_VERSION=`cut -d '.' -f 1 <<< ${HADOOP_VERSION}`
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin

# Download and install Spark
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_MAJOR_VERSION}.tgz && \
    tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_MAJOR_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_MAJOR_VERSION} /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_MAJOR_VERSION}.tgz

# Install additional jars
COPY ./additional-jars/*.jar /opt/spark/jars/

# Upgrade pip and install cmake
RUN pip install --no-cache-dir --upgrade pip setuptools cmake

# Install Python dependencies including Jupyter
RUN pip install --no-cache-dir \
    pyspark==${SPARK_VERSION} \
    jupyter \
    jupyterlab \
    ipykernel \
    torchvision \
    torch \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    streamlit

# Set up a Jupyter kernel with auto Spark initialization
RUN python -m ipykernel install --name pyspark --display-name "PySpark" && \
    mkdir -p ~/.ipython/profile_default/startup/

# Create startup script for auto Spark initialization
COPY spark_init.py ~/.ipython/profile_default/startup/00-spark-init.py

# Create a Jupyter kernel for PySpark
RUN ipython kernel install --user --name=pyspark --display-name='PySpark'

WORKDIR /workspace

CMD ["sleep", "infinity"]