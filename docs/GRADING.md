# Internal quality ratings — privacy policy

Internal quality ratings, letter grades, rubric scores, and regrade notes are **private**.

They must **not** be published in:

- `README.md` status lines or marketing summaries
- Ops / cutover / architecture status claims
- External comms, PR descriptions meant as public status, or release notes as scorecards
- Design/plan docs as success criteria that assert a letter or numeric project grade

Use factual capability language instead (what ships, what gates pass, what Downstream still owns).

Operational numeric thresholds (ECE floors, z-scores, `$` ceilings, pack sizes) are **not** grades — keep those.

Historical filenames that contain grade-like tokens (e.g. `math-9-5-design`, `prod-readiness-a-c`) may remain for link stability; their bodies must not claim public grades.
