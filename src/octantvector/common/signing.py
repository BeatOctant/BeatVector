"""The signature of a published DB version (OctantSpecs contracts/, 契約5（OctantHarvest to OctantCompass）署名).

The signed bytes are manifest.json read as JSON and canonicalised with JCS (RFC 8785), so its indentation and line
ends do not matter. The signature is Ed25519 and lives next to it in manifest.sig:

  {"algorithm": "Ed25519", "keyId": <first 16 hex of the SHA-256 of the 32-byte public key>,
   "signature": <base64 of the 64-byte signature>}

manifest.sig is never listed in the manifest's files. The public keys accepted are bundled in trusted_keys.py.

  keygen(path)          a new private key written to `path` (never overwritten) -> (public key base64, keyId)
  sign(folder, key)     manifest.sig written into `folder` with the private key file `key` -> its path
  inspect(folder, trust) who signed manifest.json -> {"status", "keyId", "detail"}; never raises. status is
                        "signed" (by a bundled key or one in trust["keys"], {keyId: public key base64}),
                        "unsigned", "malformed", "unknownKey" or "mismatch"

On the host the signature names the author; it does not block. apply_update warns (SignatureWarning) on anything but
"signed" and still uses the version: a DB is data and runs no code, and the files are checked against the
index's SHA-256 either way (2026-10-11). The author's side is stricter: build_index (generation/index.py) indexes
a version only when it is "signed", unless allow_unsigned (development).
"""
import base64
import hashlib
import json
import os
from pathlib import Path

ALGORITHM = "Ed25519"
SIG = "manifest.sig"


class SignatureWarning(UserWarning):
    """A DB version used although manifest.sig is missing, malformed, by an unknown key or does not match."""


def _canonical(folder):
    import rfc8785
    return rfc8785.dumps(json.loads((Path(folder) / "manifest.json").read_text(encoding="utf-8")))


def key_id(public_raw):
    return hashlib.sha256(public_raw).hexdigest()[:16]


def _public_raw(private):
    from cryptography.hazmat.primitives import serialization
    return private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def keygen(path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    private = Ed25519PrivateKey.generate()
    pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)  # never overwrite a key
    with os.fdopen(fd, "wb") as f:
        f.write(pem)
    raw = _public_raw(private)
    return base64.b64encode(raw).decode(), key_id(raw)


def sign(folder, key_path):
    from cryptography.hazmat.primitives import serialization
    private = serialization.load_pem_private_key(Path(key_path).read_bytes(), password=None)
    sig = {"algorithm": ALGORITHM, "keyId": key_id(_public_raw(private)),
           "signature": base64.b64encode(private.sign(_canonical(folder))).decode()}
    p = Path(folder) / SIG
    p.write_text(json.dumps(sig, indent=1) + "\n")
    return p


def check_trust(trust):
    """-> (keys, allow_unsigned) from apply_update's `trust` (None: the bundled keys only)."""
    from . import trusted_keys
    t = dict(trust or {})
    unknown = set(t) - {"keys", "allowUnsigned"}
    if unknown:
        raise ValueError(f"trust: unknown keys {', '.join(sorted(unknown))}")
    extra = dict(t.get("keys") or {})
    for kid, pub in extra.items():
        try:
            raw = base64.b64decode(pub, validate=True)
        except (ValueError, TypeError):
            raise ValueError(f"trust: key {kid} is not base64") from None
        if len(raw) != 32 or key_id(raw) != kid:
            raise ValueError(f"trust: key {kid} is not a 32-byte Ed25519 public key with that keyId")
    return {**trusted_keys.KEYS, **extra}, bool(t.get("allowUnsigned", False))


def inspect(folder, trust=None):
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    keys, _ = check_trust(trust)
    p = Path(folder) / SIG

    def out(status, detail, kid=None):
        return {"status": status, "keyId": kid, "detail": detail}
    try:
        sig = json.loads(p.read_text())
        alg, kid, raw = sig["algorithm"], sig["keyId"], base64.b64decode(sig["signature"], validate=True)
    except FileNotFoundError:
        return out("unsigned", f"{folder}: no {SIG}")
    except (OSError, ValueError, KeyError, TypeError) as ex:
        return out("malformed", f"{p}: not a signature ({ex})")
    if alg != ALGORITHM:
        return out("malformed", f"{p}: algorithm {alg!r}, only {ALGORITHM} is accepted", kid)
    if kid not in keys:
        return out("unknownKey", f"{p}: signed by key {kid}, which this OctantVector does not know", kid)
    try:
        Ed25519PublicKey.from_public_bytes(base64.b64decode(keys[kid])).verify(raw, _canonical(folder))
    except InvalidSignature:
        return out("mismatch", f"{p}: the signature does not match manifest.json", kid)
    return out("signed", f"{p}: signed by key {kid}", kid)

