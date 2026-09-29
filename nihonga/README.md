# Nihonga research record

`index.html` follows the site's shared style. `results.json` contains the precise
values behind the tables and graphs, plus SHA-256 hashes of the saved source
artifacts. The five gallery sheets are existing Phase 3 identity-bypass comparisons,
not outputs from the pretrained or healed bridge model.

To refresh from the research checkout (no GPU or network calls):

```powershell
python tools/build_nihonga.py --source ../nihonga
```

The script publishes only explicitly selected summary fields and comparison sheets.
Update the date and narrative when new experiments change the findings. Keep failures
and limitations visible, distinguish hidden-fit losses from velocity errors and image
quality, and never infer a speedup from the number of removed blocks alone.

The production site is deployed by GitHub Pages from `main`, at `/nihonga/`.

The Nihonga hero is an original AI-generated illustration, not a student-model
sample or an antique reproduction. The built-in image-generation tool created it
to match the existing monochrome engraved plates. Its source is
`assets/nihonga.png`; JPEG and WebP encodings are used on the page. The generation
prompt is preserved in `assets/nihonga-prompt.txt`.
