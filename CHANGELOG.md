# 2021.09.28

- Updated terms of service.
  - Reworded a few sections.
  - Replaced uppercase from sections 5 to 7 with highlighting.

# 2021.09.26

- Rewrote legal content.
  - Updated terms of service to be more like [Legalmattic](https://github.com/Automattic/legalmattic)'s terms of
    service.
    - Licensed terms of service as [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) as required.
  - Made privacy policy more specific.
  - Updated attribution section to use a more relevant "hugging" emoji as opposed to a generic copyright symbol emoji.
    - Also renamed attribution section to "Attribution" instead of "Emoji Disclaimer" as we now have to provide
      attribution for a second, unrelated party.
- Made minor clarifications within docstrings in the model implementation file.
  - Specifically, referred to the sample as a "non-jagged list of lists" rather than a "2D rectangular list".
- Changed listed model statistics to be more representative of the average fitting result.
  - Previously, the output from a single run was used. This update enables flexibility in refitting without having to
    constantly update the listed model statistics for minor fluctuations.
  - The fact that the statistics are only approximations has been noted in the new terms of service.
- Added `prettier-ignore` flags throughout several pages.
  - Although ugly, the flags enable the use of Prettier without messing up the Jinja syntax.

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
