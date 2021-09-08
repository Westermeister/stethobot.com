"""Flask entry point."""

from flask import Flask, render_template, send_file

app = Flask(__name__)


@app.route("/robots.txt")
def robots():
    """Return robots.txt."""
    return send_file("robots.txt")


@app.route("/sitemap.xml")
def sitemap():
    """Return sitemap.xml."""
    return send_file("sitemap.xml")


@app.route("/")
def index():
    """Render the homepage."""
    return render_template("index.html")


@app.route("/legal")
def legal():
    """Render the legal page."""
    return render_template("legal.html")


@app.route("/conditions/")
def conditions():
    """Render the index of conditions."""
    return render_template("conditions/index.html")


@app.route("/conditions/breast-cancer")
def breast_cancer():
    """Render the breast cancer input form."""
    return render_template("conditions/breast-cancer.html")


@app.route("/conditions/diabetes")
def diabetes():
    """Render the diabetes input form."""
    return render_template("conditions/diabetes.html")


@app.route("/conditions/heart-disease")
def heart_disease():
    """Render the heart disease input form."""
    return render_template("conditions/heart-disease.html")


@app.route("/api/diagnose/breast-cancer")
def breast_cancer_diagnosis():
    """Return the diagnosis for breast cancer."""
    return  # TODO


@app.route("/api/diagnosis/diabetes")
def diabetes_diagnosis():
    """Return the diagnosis for diabetes."""
    return  # TODO


@app.route("/api/diagnosis/heart-disease")
def heart_disease_diagnosis():
    return  # TODO


@app.route("/docs/breast-cancer")
def breast_cancer_docs():
    """Return the documentation for breast cancer dataset."""
    return send_file("./static/docs/breast-cancer-docs.pdf")
