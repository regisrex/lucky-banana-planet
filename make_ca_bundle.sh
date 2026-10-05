#!/usr/bin/env bash
# listen.rba.co.rw serves only its leaf certificate and omits the RapidSSL
# intermediate. curl on macOS fetches the issuer via AIA and so appears to work;
# ffmpeg/OpenSSL and Python requests do not, and fail verification. This builds a
# bundle of the normal public roots PLUS that one missing intermediate, so we keep
# certificate verification ON instead of disabling it.
set -euo pipefail
cd "$(dirname "$0")"
AIA="http://cacerts.rapidssl.com/RapidSSLTLSRSACAG1.crt"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

curl -fsS -o "$tmp/i.crt" "$AIA"
openssl x509 -inform DER -in "$tmp/i.crt" -out "$tmp/i.pem" 2>/dev/null \
  || cp "$tmp/i.crt" "$tmp/i.pem"

if python3 -c "import certifi" 2>/dev/null; then
  python3 -c "import certifi;print(open(certifi.where()).read())" > rba-ca-bundle.pem
else
  cat /opt/homebrew/etc/openssl@3/cert.pem /etc/ssl/cert.pem 2>/dev/null > rba-ca-bundle.pem
fi
cat "$tmp/i.pem" >> rba-ca-bundle.pem

echo "wrote rba-ca-bundle.pem ($(grep -c 'BEGIN CERTIFICATE' rba-ca-bundle.pem) certs)"
openssl x509 -in "$tmp/i.pem" -noout -subject
echo
echo "leaf cert expiry (RBA must renew this):"
echo | openssl s_client -connect listen.rba.co.rw:8008 -servername listen.rba.co.rw 2>/dev/null \
  | openssl x509 -noout -dates 2>/dev/null || true
