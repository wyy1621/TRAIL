"""Message-to-triplet helper relocated verbatim from pipeline/eval_utils.py."""
from common import AsPathDataClean, build_consecutive_triplets

def extract_clean_triplets(as_path_str: str, peer_as=None, ixp_asns: set[int] | frozenset[int] | None = None) -> set[tuple[int, int, int]]:
    """Extract all consecutive RADBench message triplets.

    Args:
        as_path_str: Raw AS_PATH string from a BGP4MP announcement.
        peer_as: Unused BGP4MP peer AS field, kept for call-site compatibility.
            Message-to-triplet matching always uses all consecutive triplets.
        ixp_asns: IXP / Route Server ASN set removed before triplet extraction.

    Returns:
        Set of all consecutive triplets from the cleaned AS path.
    """
    clean_as_path = AsPathDataClean(as_path_str.split(), ixp_asns=ixp_asns)
    if not clean_as_path or len(clean_as_path) < 3:
        return set()
    return build_consecutive_triplets(clean_as_path)
