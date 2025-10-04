# Sales Force API Dockerfile.
#
#Copyright 2025 Salesforce Inc.
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

# Set PYTHONPATH environment variable
ENV PYTHONPATH="/usr/lib/python3.9/site-packages"

# pyodbc dependencies
RUN apt-get update && \
  apt-get install -y gcc g++ libgssapi-krb5-2 curl jq && \
  apt-get clean all

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


#RUN pip install --no-cache-dir -r /src/requirements.txt
RUN rm -r ~/.cache/pip/selfcheck/


# -----------------------------------------------------------------------------------------------------------------------------------------
FROM base AS production

# Switch to the non-privileged user to run the application.
USER appuser

COPY ./src ./src
EXPOSE 8000
CMD ["uvicorn", "src.main:api", "--host", "0.0.0.0", "--port", "8000"]

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
