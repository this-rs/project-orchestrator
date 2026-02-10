#!/bin/sh
# Project Orchestrator installer
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/this-rs/project-orchestrator/main/install.sh | sh
#
# Options (via env vars or flags):
#   --version <version>    Install a specific version (default: latest)
#   --no-frontend          Install the light variant (no embedded frontend)
#   --install-dir <path>   Installation directory (default: ~/.local/bin)
#   --help                 Show this help message

set -eu

REPO="this-rs/project-orchestrator"
BINARY_NAME="orchestrator"
DEFAULT_INSTALL_DIR="$HOME/.local/bin"

# --- Defaults ---
VERSION=""
VARIANT="full"
INSTALL_DIR="${INSTALL_DIR:-$DEFAULT_INSTALL_DIR}"

# --- Colors (only if terminal) ---
if [ -t 1 ]; then
  BOLD="\033[1m"
  GREEN="\033[32m"
  YELLOW="\033[33m"
  RED="\033[31m"
  RESET="\033[0m"
else
  BOLD="" GREEN="" YELLOW="" RED="" RESET=""
fi

info()  { printf "${GREEN}info${RESET}: %s\n" "$1"; }
warn()  { printf "${YELLOW}warn${RESET}: %s\n" "$1"; }
error() { printf "${RED}error${RESET}: %s\n" "$1" >&2; exit 1; }

# --- Parse arguments ---
while [ $# -gt 0 ]; do
  case "$1" in
    --version)
      VERSION="$2"
      shift 2
      ;;
    --no-frontend)
      VARIANT="light"
      shift
      ;;
    --install-dir)
      INSTALL_DIR="$2"
      shift 2
      ;;
    --help|-h)
      printf "%s\n" "Project Orchestrator installer"
      printf "\n"
      printf "Usage: install.sh [OPTIONS]\n"
      printf "\n"
      printf "Options:\n"
      printf "  --version <version>    Install a specific version (default: latest)\n"
      printf "  --no-frontend          Install the light variant (no embedded frontend)\n"
      printf "  --install-dir <path>   Installation directory (default: ~/.local/bin)\n"
      printf "  --help                 Show this help message\n"
      exit 0
      ;;
    *)
      error "Unknown option: $1 (use --help for usage)"
      ;;
  esac
done

# --- Detect OS ---
detect_os() {
  case "$(uname -s)" in
    Darwin) echo "macos" ;;
    Linux)  echo "linux" ;;
    *)      error "Unsupported operating system: $(uname -s). Use install.ps1 for Windows." ;;
  esac
}

# --- Detect architecture ---
detect_arch() {
  case "$(uname -m)" in
    x86_64|amd64)   echo "x86_64" ;;
    aarch64|arm64)   echo "arm64" ;;
    *)               error "Unsupported architecture: $(uname -m)" ;;
  esac
}

# --- Check for required tools ---
check_deps() {
  for cmd in curl tar; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      error "Required command not found: $cmd"
    fi
  done

  # sha256sum or shasum (macOS)
  if command -v sha256sum >/dev/null 2>&1; then
    SHA256_CMD="sha256sum"
  elif command -v shasum >/dev/null 2>&1; then
    SHA256_CMD="shasum -a 256"
  else
    warn "Neither sha256sum nor shasum found — skipping checksum verification"
    SHA256_CMD=""
  fi
}

# --- Get latest version from GitHub API ---
get_latest_version() {
  local latest
  latest=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name"' | sed -E 's/.*"tag_name": *"v([^"]+)".*/\1/')
  if [ -z "$latest" ]; then
    error "Failed to determine latest version from GitHub API"
  fi
  echo "$latest"
}

# --- Main ---
main() {
  check_deps

  OS=$(detect_os)
  ARCH=$(detect_arch)

  info "Detected platform: ${OS}-${ARCH}"

  # Resolve version
  if [ -z "$VERSION" ]; then
    info "Fetching latest version..."
    VERSION=$(get_latest_version)
  fi
  info "Version: v${VERSION}"

  # Build archive name
  if [ "$VARIANT" = "full" ]; then
    ARCHIVE_NAME="${BINARY_NAME}-full-${VERSION}-${OS}-${ARCH}"
  else
    ARCHIVE_NAME="${BINARY_NAME}-${VERSION}-${OS}-${ARCH}"
  fi
  ARCHIVE_FILE="${ARCHIVE_NAME}.tar.gz"

  info "Variant: ${VARIANT} (${ARCHIVE_FILE})"

  # Download URLs
  BASE_URL="https://github.com/${REPO}/releases/download/v${VERSION}"
  ARCHIVE_URL="${BASE_URL}/${ARCHIVE_FILE}"
  CHECKSUMS_URL="${BASE_URL}/checksums-sha256.txt"

  # Create temp directory
  TMP_DIR=$(mktemp -d)
  trap 'rm -rf "$TMP_DIR"' EXIT

  # Download archive
  info "Downloading ${ARCHIVE_URL}..."
  if ! curl -fSL --progress-bar -o "${TMP_DIR}/${ARCHIVE_FILE}" "$ARCHIVE_URL"; then
    error "Failed to download archive. Check that version v${VERSION} exists and has a ${OS}-${ARCH} build."
  fi

  # Verify checksum
  if [ -n "$SHA256_CMD" ]; then
    info "Verifying checksum..."
    if curl -fsSL -o "${TMP_DIR}/checksums-sha256.txt" "$CHECKSUMS_URL"; then
      EXPECTED=$(grep "$ARCHIVE_FILE" "${TMP_DIR}/checksums-sha256.txt" | awk '{print $1}')
      if [ -n "$EXPECTED" ]; then
        ACTUAL=$($SHA256_CMD "${TMP_DIR}/${ARCHIVE_FILE}" | awk '{print $1}')
        if [ "$EXPECTED" != "$ACTUAL" ]; then
          error "Checksum mismatch!\n  Expected: ${EXPECTED}\n  Actual:   ${ACTUAL}"
        fi
        info "Checksum verified"
      else
        warn "Archive not found in checksums file — skipping verification"
      fi
    else
      warn "Could not download checksums — skipping verification"
    fi
  fi

  # Extract
  info "Extracting..."
  tar xzf "${TMP_DIR}/${ARCHIVE_FILE}" -C "$TMP_DIR"

  # Install
  mkdir -p "$INSTALL_DIR"
  cp "${TMP_DIR}/${ARCHIVE_NAME}/${BINARY_NAME}" "${INSTALL_DIR}/${BINARY_NAME}"
  chmod +x "${INSTALL_DIR}/${BINARY_NAME}"

  info "Installed ${BOLD}${BINARY_NAME}${RESET} to ${INSTALL_DIR}/${BINARY_NAME}"

  # Check if install dir is in PATH
  case ":$PATH:" in
    *":${INSTALL_DIR}:"*)
      ;;
    *)
      warn "${INSTALL_DIR} is not in your PATH"
      printf "\n  Add this to your shell profile:\n"
      printf "    export PATH=\"%s:\$PATH\"\n\n" "$INSTALL_DIR"
      ;;
  esac

  # Verify installation
  if command -v "$BINARY_NAME" >/dev/null 2>&1; then
    info "Installation complete! Run '${BINARY_NAME} --help' to get started."
  else
    info "Installation complete! You may need to restart your shell or update your PATH."
  fi
}

main
