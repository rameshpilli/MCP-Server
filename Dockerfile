# Use Python 3.11 slim image
FROM innersource-docker.artifactory.fg.rbc.com/container-hub/python:3.11-slim
# FROM innersource-docker.rbcartifactory.fg.rbc.com/container-hub/python:3.11

USER root

# Copy certificates
COPY artifacts/Production_RBC_G2_Root_CA.cer /usr/local/share/ca-certificates/root.crt 
COPY artifacts/Production_RBC_G2_Root_CA.cer /etc/pki/ca-trust/source/anchors/
COPY artifacts/Production_RBC_G2_Intermediate_CA.cer /usr/local/share/ca-certificates/intermediate.crt 
COPY artifacts/Production_RBC_G2_Intermediate_CA.cer /etc/pki/ca-trust/source/anchors/
COPY artifacts/rbc-bundle.pem /rbc-bundle.pem 
RUN chmod 777 /rbc-bundle.pem

# Configure pip to use RBC's artifact repository
COPY artifacts/pip.conf /pip.conf
ENV PIP_CONFIG_FILE /pip.conf

# Set certificate environment variables
ENV REQUESTS_CA_BUNDLE /rbc-bundle.pem 
ENV SSL_CERT_FILE /rbc-bundle.pem 
ENV CURL_CA_BUNDLE /rbc-bundle.pem

# Update certificate trust stores - use the appropriate command for your base image
# Try update-ca-certificates (Debian/Ubuntu) instead of update-ca-trust (RHEL/CentOS)
RUN if command -v update-ca-certificates > /dev/null; then \
        update-ca-certificates; \
    elif command -v update-ca-trust > /dev/null; then \
        update-ca-trust; \
    else \
        echo "Warning: Neither update-ca-certificates nor update-ca-trust is available"; \
    fi

# Set up application directory
WORKDIR /usr/src/elements-ai-server

# Copy application code
COPY app/ ./app/
COPY docs/ ./docs/
COPY tests/ ./tests/
COPY ui/ ./ui/
COPY run.py .
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip

RUN pip install --no-index --find-links=./artifacts/compass-sdk compass_sdk

RUN pip install -r requirements.txt

# Expose ports for API, MCP server, and UI
EXPOSE 8000 8001 8501

# Set environment variables
ENV IN_KUBERNETES="true"
ENV LOG_TO_STDOUT="true"

# Start application
CMD ["python", "run.py"] 