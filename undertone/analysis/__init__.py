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
    arm_direction,
    cost_status,
    l3_failure_reason,
    prominence_2x2,
    sanity_checks,
    scorer_gap,
    sign_test,
    table1_main,
    table1_nulls,
    table2_ladder,
    table4_truncation,
    signatures,
    table_language,
    usable,
)

__all__ = [
    "MixedHardware", "signatures", "cost_status", "l3_failure_reason",
    "sanity_checks", "scorer_gap", "sign_test", "prominence_2x2", "arm_direction", "table1_main", "table1_nulls",
    "table2_ladder", "table4_truncation", "table_language", "usable",
    "all_figures", "fig2_signature", "fig3_fingerprint", "fig4_ladder",
    "fig5_repetition",
]
