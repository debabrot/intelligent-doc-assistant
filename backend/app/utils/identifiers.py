import hashlib


def generate_deterministic_id(content: str, metadata: dict) -> str:
    """Generate a deterministic ID based on content and metadata."""
    # Sort metadata keys to ensure consistent string representation
    metadata_str = str(sorted(metadata.items()))
    combined = content + metadata_str
    hash_obj = hashlib.sha256(combined.encode('utf-8'))
    # Use first 32 chars of hex digest as ID (shorter, still very unique)
    return hash_obj.hexdigest()[:32]
