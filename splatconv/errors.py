"""Exceptions raised across the reading/processing pipeline.

Each carries a user-facing `message` (in French, since the tool targets a
French-speaking geomatics audience) suitable for display as-is in the CLI or
web UI, separate from any lower-level technical detail.
"""


class SplatConvError(Exception):
    """Base class for all expected/handled errors in the pipeline."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class UnsupportedFormatError(SplatConvError):
    """Raised when the input file extension is not one we know how to read."""


class RcpNotSupportedError(SplatConvError):
    """Raised when a .rcp file is uploaded.

    Autodesk ReCap's .rcp/.rcs format is a proprietary, undocumented binary
    format with no reliable open-source parser. Rather than guessing at a
    binary layout, we ask the user to re-export from ReCap Pro.
    """

    def __init__(self, path: str):
        message = (
            "Le format .rcp (Autodesk ReCap) est un format propriétaire non "
            "documenté et ne peut pas être lu directement.\n\n"
            "Pour continuer :\n"
            "1. Ouvrez ce projet dans Autodesk ReCap Pro.\n"
            "2. Menu Export → E57 (recommandé) — ou PTS/XYZ/LAS.\n"
            "3. Importez ici le fichier exporté (.e57, .pts, .xyz ou .las) "
            "à la place du .rcp."
        )
        super().__init__(message)
        self.path = path


class CorruptFileError(SplatConvError):
    """Raised when a file matches a known extension but fails to parse."""


class EmptyPointCloudError(SplatConvError):
    """Raised when a source file yields zero usable points."""


class InsufficientMemoryError(SplatConvError):
    """Raised (pre-emptively) when a cloud is judged too large to process safely."""
