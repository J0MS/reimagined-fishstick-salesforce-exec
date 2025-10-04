# Random Control Trial API Dockerfile.
#
#Copyright 2024 Anheuser Busch InBev Inc.
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

ARG PYTHON_VERSION=3.9.20
FROM python:${PYTHON_VERSION}-slim as base

# Create a non-privileged user that the app will run under.
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/src" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser
RUN usermod -aG appuser appuser

# Change the ownership of the working directory to the non-root user
RUN chown -R appuser:appuser /src

# Install OpenJDK-17
# RUN add-apt-repository ppa:openjdk-r/ppa
#RUN apt-get update && \
#    apt-get install -y openjdk-17-jre && \
#    apt-get clean;
# Set the JAVA_HOME environment variable
#ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk-amd64
#ENV PATH $JAVA_HOME/bin:$PATH

# Install OpenJDK 17 (compatible with PySpark 3.3.0)
RUN apt-get update && \
    apt-get install -y openjdk-17-jre && \
    apt-get clean;

# Set JAVA_HOME environment variable explicitly
#ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
#ENV PATH="$JAVA_HOME/bin:$PATH"

# Dynamically set JAVA_HOME
RUN export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java)))) && \
    echo "JAVA_HOME is set to $JAVA_HOME"

# Verify Java installation
RUN java -version

# Set PYTHONPATH environment variable
ENV PYTHONPATH="/usr/lib/python3.9/site-packages"
ENV PYSPARK_SUBMIT_ARGS "--master local[*] --executor-memory 16g --driver-memory 16g pyspark-shell"
#ENV PYSPARK_SUBMIT_ARGS="--master local[2] pyspark-shell"

# pyodbc dependencies
RUN apt-get update && \
  apt-get install -y gcc g++ libgssapi-krb5-2 curl jq && \
  apt-get clean all

#Installing Pyspark connectors
RUN printf "INFO: Installing Pyspark connectors \n"

WORKDIR /src
COPY ./requirements.txt ./requirements.txt
RUN pip install --upgrade pip


# Download dependencies as a separate step to take advantage of 
# Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
   --mount=type=bind,source=requirements.txt,target=requirements.txt \
   python -m pip install -r requirements.txt

# Test PySpark with a simple command
RUN python -c "from pyspark.sql import SparkSession; spark = SparkSession.builder.getOrCreate(); print(spark.version)"

#RUN pip install --no-cache-dir -r /src/requirements.txt
RUN rm -r ~/.cache/pip/selfcheck/


# Setting  build arguments
ARG LOCAL_PACKAGE
ARG JFROG_CONNECTION_STRING

# Loading build arguments from .env file
RUN echo "JFROG_CREDENTIALS = $JFROG_CREDENTIALS"

# If LOCAL_PACKAGE build argument was provided, copy the whl file
COPY $LOCAL_PACKAGE ./

# Determine the origin of testops_tools_random_control_trial_2 package (remote or local)
RUN if [ -z "$LOCAL_PACKAGE" ]; then \
        printf "INFO: Installing remote built package \n" && \
        # Building JFROG_CONNECTION_STRING, loading JFROG_CREDENTIALS from .env file
        #JFROG_CONNECTION_STRING=https://$JFROG_CREDENTIALS@abinbev.jfrog.io/artifactory/api/pypi/abia-local-light-house/simple && \
        # Building the testops_query file
        echo 'items.find({"@pypi.name":{"$eq":"testops-tools-random-control-trial-2"}}).sort({"$asc": ["updated"]})' > testops_query.aql && \
        # Sending request to JFrog platform to get the latest package path (like package/version)
        #latest_package_path=$(curl -u $JFROG_CREDENTIALS -X POST https://abinbev.jfrog.io/artifactory/api/search/aql -H "content-type: text/plain" -d @testops_query.aql | jq -r '.results[-1].path'); latest_package_version="${latest_package_path#*/}"; echo $latest_package_version > latest_package_version && \
        # Capturing the latest package version
        printf "INFO: Latest package version=$(cat latest_package_version) \n" && \
        # Installing JFrog repository
        pip install --no-cache-dir --index-url $JFROG_CONNECTION_STRING --extra-index-url https://pypi.org/simple testops_tools_random_control_trial_2==0.8.4 testops_tools_outliers_treatment==0.8.4; \
    else \
        printf "INFO: Installing locally built package $LOCAL_PACKAGE \n" && pip3 install $LOCAL_PACKAGE; \
    fi

# Create folder to save celery logs
RUN mkdir -p celery_logs

# Change the ownership of the celery_logs directory to the non-root user
RUN chown -R appuser:appuser /src/celery_logs

# -----------------------------------------------------------------------------------------------------------------------------------------
FROM base AS production

# Switch to the non-privileged user to run the application.
USER appuser

COPY ./src ./src
EXPOSE 8000
CMD ["uvicorn", "src.main:rct_api", "--host", "0.0.0.0", "--port", "8000"]

# -----------------------------------------------------------------------------------------------------------------------------------------

FROM base as testing
#WORKDIR /src
ARG PYTHON_VERSION=3.9
WORKDIR /

# Install testing suite
RUN pip install  httpx pytest pytest-dotenv pytest-cov pytest-xdist pytest-mock
RUN printf "INFO: Testing suite installed \n"

COPY ./tests/ src/tests/
COPY ./src src/src

# -----------------------------------------------------------------------------------------------------------------------------------------
FROM testing as unit_test

# Switch to the privileged user to run write test reports
USER root
#RUN setfacl -R -m u:appuser:rwx /src/

#CMD  bash -c "pytest --cov-config=.coveragerc src/tests/unit_test --junitxml=reports/unittest-${PYTHON_VERSION}.xml --cov-report=xml:reports/coverage-${PYTHON_VERSION}.xml --cov-report=html:reports/coverage_report-${PYTHON_VERSION} --cov=src"

CMD bash -c "pytest --cov-config=.coveragerc src/tests/unit_test \
--junitxml=reports/unittest-${PYTHON_VERSION}.xml \
--junitxml=reports/pytest-${PYTHON_VERSION}.xml \
--cov-report=xml:reports/coverage-${PYTHON_VERSION}.xml \
--cov-report=html:reports/coverage_report-${PYTHON_VERSION}.html \
--cov-report=term-missing --cov=src | tee reports/pytest-coverage-${PYTHON_VERSION}.txt"

# -----------------------------------------------------------------------------------------------------------------------------------------
FROM testing as unit_test_coverage
# Switch to the privileged user to run write test reports
USER root

CMD  bash -c "pytest --cov-config=.coveragerc src/tests/unit_test --junitxml=reports/pytest-${PYTHON_VERSION}.xml --cov-report term-missing  --cov=src | tee reports/pytest-coverage-${PYTHON_VERSION}.txt"

# -----------------------------------------------------------------------------------------------------------------------------------------
FROM testing as integration_test
# Switch to the privileged user to run the application.
USER root

CMD  bash -c "pytest --cov-config=.coveragerc src/tests/integration_test --junitxml=reports/pytest-${PYTHON_VERSION}.xml --cov-report term-missing  --cov=src | tee reports/pytest-coverage-${PYTHON_VERSION}.txt"

# -----------------------------------------------------------------------------------------------------------------------------------------
