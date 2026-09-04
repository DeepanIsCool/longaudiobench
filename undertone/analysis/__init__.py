"""Paper tables and figures. No composite scores, by design."""

from .figures import (
    all_figures,
    fig2_signature,
    fig3_fingerprint,
    fig4_ladder,
    fig5_repetition,
)
from .tables import (
    MixedHardware,
    cost_status,
    sanity_checks,
    scorer_gap,
    table1_main,
    table1_nulls,
    table2_ladder,
    table4_truncation,
    signatures,
    table_language,
    usable,
)

__all__ = [
    "MixedHardware", "signatures", "cost_status",
    "sanity_checks", "scorer_gap", "table1_main", "table1_nulls",
    "table2_ladder", "table4_truncation", "table_language", "usable",
    "all_figures", "fig2_signature", "fig3_fingerprint", "fig4_ladder",
    "fig5_repetition",
]
