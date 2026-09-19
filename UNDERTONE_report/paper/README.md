# Paper build

    python make_numbers.py      # data/ -> numbers.tex (every number in the paper)
    tectonic siren.tex          # -> siren.pdf   (pdflatex siren.tex x2 also works)

`siren.tex` states no item or dataset count; counts appear only in table
columns and all come from `numbers.tex`. After a new run: re-bank, run
`scripts/consolidate_report_data.py`, then the two lines above.

Macros to set by hand (top of siren.tex): `\bench`, `\corpus`,
`\plannedcorpus`, and the author block.
