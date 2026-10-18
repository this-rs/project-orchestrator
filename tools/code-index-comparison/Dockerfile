FROM debian:trixie-slim
ARG TARGETARCH
ARG PO_VERSION=0.0.17
ARG CBM_VERSION=0.11.0
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git python3 python3-pip python3-venv libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
RUN set -eu; \
    case "$TARGETARCH" in amd64) po_arch=x86_64;; arm64) po_arch=arm64;; *) exit 1;; esac; \
    mkdir -p /tmp/po /tmp/cbm; \
    po_archive="orchestrator-${PO_VERSION}-linux-${po_arch}.tar.gz"; \
    curl -fsSL --retry 3 "https://github.com/this-rs/project-orchestrator/releases/download/v${PO_VERSION}/${po_archive}" -o "/tmp/${po_archive}"; \
    curl -fsSL --retry 3 "https://github.com/this-rs/project-orchestrator/releases/download/v${PO_VERSION}/checksums-sha256.txt" -o /tmp/po-checksums; \
    cd /tmp; grep "  ${po_archive}$" po-checksums | sha256sum -c -; \
    tar xzf "/tmp/${po_archive}" -C /tmp/po; \
    cp "/tmp/po/orchestrator-${PO_VERSION}-linux-${po_arch}/orchestrator" /usr/local/bin/orchestrator; \
    cp "/tmp/po/orchestrator-${PO_VERSION}-linux-${po_arch}/mcp_server" /usr/local/bin/po-mcp; \
    cbm_archive="codebase-memory-mcp-linux-${TARGETARCH}.tar.gz"; \
    curl -fsSL --retry 3 "https://github.com/DeusData/codebase-memory-mcp/releases/download/v${CBM_VERSION}/${cbm_archive}" -o "/tmp/${cbm_archive}"; \
    curl -fsSL --retry 3 "https://github.com/DeusData/codebase-memory-mcp/releases/download/v${CBM_VERSION}/checksums.txt" -o /tmp/cbm-checksums; \
    grep "  ${cbm_archive}$" cbm-checksums | sha256sum -c -; \
    tar xzf "/tmp/${cbm_archive}" -C /tmp/cbm; \
    cp /tmp/cbm/codebase-memory-mcp /usr/local/bin/codebase-memory-mcp; \
    chmod 755 /usr/local/bin/orchestrator /usr/local/bin/po-mcp /usr/local/bin/codebase-memory-mcp; \
    rm -rf /tmp/po /tmp/cbm /tmp/*checksums* /tmp/*.tar.gz
COPY reader /opt/reader
RUN python3 -m pip install --break-system-packages --target /opt/reader/python -r /opt/reader/requirements.txt
WORKDIR /app
# This exact bind-mounted checkout is created and pinned by prepare.py.
RUN git config --system --add safe.directory /workspace/po
CMD ["sleep", "infinity"]
