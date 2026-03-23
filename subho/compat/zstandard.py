"""
Minimal compatibility stub for environments that need to import
`oxDNA_analysis_tools` on plain-text trajectories without the real `zstandard`
package installed.

This file is only intended for the local output_bonds fallback path used by
`subho/compare_single_frame_to_oat.py`. If compressed trajectory support is
needed, install the real `zstandard` package instead.
"""


class _UnavailableZstd:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "The real `zstandard` package is not installed. "
            "This compatibility stub only supports workflows that never touch "
            "compressed trajectories."
        )


class ZstdDecompressor(_UnavailableZstd):
    pass


class ZstdCompressor(_UnavailableZstd):
    pass
