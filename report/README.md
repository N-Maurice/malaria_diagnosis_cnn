# Final project report

- **Final_Project_Report.docx** — editable Word report with populated automatic
  contents, figure/table lists, heading styles, caption fields and page numbers.
- **Final_Project_Report.pdf** — rendered review copy. The assignment lists PDF
  as the submission format.
- **figures/** — actual notebook outputs, source-derived workflow/architecture
  diagrams and graphs based on recorded data. Original high-resolution panels
  are retained where useful.
- **supporting_materials/** — fixed evidence snapshots, derived metric tables,
  source hashes, assignment text, test output and the document builder.
- **drafts/** and **references.bib** — existing team material preserved.

## Evidence and length

The concise report is **15 pages total, including the title page, contents,
caption lists, references and appendices**. There are 7 numbered figures,
11 numbered tables, three populated navigation indexes and nine external
references, including seven scholarly sources. The Word archive was validated,
reopened successfully and rendered with LibreOffice for visual review.

The report describes David's eight recorded experiments and Sarah's seven
recorded experiments. Maurice's baseline is explicitly a **four-epoch fixed
snapshot of a partial run**, not a final result. Laura's missing implementation
is disclosed. Later training does not automatically update this document.

Sarah's split and metric conflicts are explained in the main text and Appendix B.
The test comparison consistently uses the explicit saved notebook confusion
counts; derived metrics are separately preserved as CSV/JSON. No experiment was
run to manufacture missing report results. All 11 existing synthetic tests passed
when checked during report preparation.

## Before submission

Fill the labelled institutional-details and external-notebook/log-link gaps.
Confirm each owner's selected checkpoint, metrics and contribution. Replace
missing model/evaluation placeholders only with completed, traceable evidence.
Add high-resolution TensorBoard dashboard comparisons and open-access raw-log
links as required by the PDF. The report's Appendix B maps requirements to
sections and identifies unfinished deliverables.

The assignment requires original student narrative with below 20% AI-generated
content. This report is AI-assisted and **does not certify that requirement**.
The team must review and revise the explanations and interpretations in its own
words, following the assignment policy, before submission.

## Editing and rebuilding

In Word, use **Ctrl+A → F9** after editing to update navigation/caption fields.
Changes in layout can alter the page count; re-export and check the PDF.

`supporting_materials/build_report.py` recreates the document from the fixed
snapshots and saved figure files using `python-docx`. It overwrites the Word
file and requires fields to be updated again in Word or LibreOffice. Preserve
an edited student-authored copy before using the builder. It does not train any
model, refresh snapshots or replace the existing team draft.
