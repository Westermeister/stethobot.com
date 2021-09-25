# 2021.09.24

- Made minor naming and formatting adjustments within `api` directory.
  - Added missing newline at end of diabetes' sample file.
  - Added `.csv` suffix to all sample files, and updated each `fit.py` accordingly.
  - Removed unicode characters (em dashes) from each `README.txt` file.

# 2021.09.23

- Migrated from old repository to new one.
- Refactored frontend.
  - Made several quality adjustments to homepage.
    - Homepage is more compact, featuring a much smaller hero section.
    - Adjusted wording of descriptions to emphasize that Stethobot is purely a proof-of-concept.
  - Updated index of conditions.
    - Changed name of `DIAGNOSE` navbar link to `CONDITIONS` link.
    - Index is not represented by `/conditions/` instead of an explicit file `/conditions.html`.
  - Updated input forms for condition diagnosis.
    - All React forms have been replaced with pure HTML forms powered by a Flask backend.
    - Removed honeypot field from all forms.
    - Removed complicated and useless placeholder inputs from `/conditions/breast-cancer`.
  - Added confidence score from 0 to 100 to diagnosis page.
  - Removed contact page and associated navbar link.
  - Added copyright notice to all pages' footers.
  - Rescaled emojis to fit better within text.
- Refactored backend.
  - Backend now uses Flask instead of Node.
    - Moved all pages to server-side rendering instead of serving static files.
    - Replaced the Node implementation of the diagnosis API with Flask.
    - Removed the Node implementation of the contact API as it's no longer needed.
  - Added `robots.txt` and `sitemap.xml` for SEO.
  - Refactored implementation and interface of the model.
    - Updated tests as needed.
