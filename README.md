# stethobot.com

This is the repository for [Stethobot](https://stethobot.com/).

In the root, you'll find configuration files for prettier and git. You'll also find a list of dependencies within the
`package.json` and `requirements.txt` files, as well as a `Makefile` for building the project. Finally, there's a
`CHANGELOG.md` for easily reading and understanding changes without having to rely on git.

You'll also find the `stethobot` and `tests` directories.

## stethobot

In the `stethobot` directory, as the name implies, you'll find the implementation code for Stethobot. This includes
things like a `sitemap.xml` and the related `robots.txt` for SEO. It also includes the entry point for Flask, which is
`app.py`. This file is responsible for responding to incoming requests with the appropriate response. For static web
pages, this means simply rendering a Flask template via Jinja. The related files for that are found in the `static` and
`templates` directories. For actually computing diagnoses, this means calling more complicated code.

That complicated code resides in the `api` subdirectory. It contains a custom implementation of a machine learning model
called a [Balanced Random Forest](https://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf). This
implementation is fully parallelized, enabling very fast training speeds, with the tradeoff that a decent amount of
memory is required. The exact amount will depend on the size of the training dataset.

Speaking of the training datasets: those can be found in each of the subdirectories of the `api` directory. In addition
to the dataset itself, each a `README.txt` documenting the dataset, as well as a script to preprocess it and train a
model for it.

After a model is trained, the script will also output statistics; namely, the estimated accuracy, sensitivity, and
specificity, which helps us understand how the model is likely to perform on real-world data.

## tests

The tests in here (made with [pytest](https://docs.pytest.org/en/6.2.x/)) not only validate that the machine learning
model is functional, but they also compare it against a reference implementation:
[scikit-learn](https://scikit-learn.org/). The famous [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) is
used for the comparison. Together, these tests ensure that what we're putting in production is actually functional.
