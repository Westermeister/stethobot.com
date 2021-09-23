# Copyright (c) 2021 The Stethobot Authors. All rights reserved.

"""Flask entry point."""

from flask import Flask, render_template, request, send_file

from api.model import DiagnosticModel

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


@app.route("/conditions/breast-cancer", methods=["GET", "POST"])
def breast_cancer():
    """Handle diagnosis requests for breast cancer."""
    if request.method == "GET":
        return render_template("conditions/breast-cancer.html")
    else:
        ORDER = [
            "radius",
            "texture",
            "perimeter",
            "area",
            "smoothness",
            "compactness",
            "concavity",
            "concave",
            "symmetry",
            "fractal",
        ]
        observation = []
        for i in range(3):
            for name in ORDER:
                csv = request.form.get(name).split(",")
                observation.append(float(csv[i]))
        model = DiagnosticModel()
        with open("./api/breast_cancer/model.json") as f:
            model.from_json(f.read())
        diagnosis, confidence_score = model.diagnose(observation)
        return render_template(
            "conditions/diagnosis.html",
            condition="breast cancer",
            diagnosis=diagnosis,
            confidence_score=confidence_score,
        )


@app.route("/conditions/diabetes", methods=["GET", "POST"])
def diabetes():
    """Handle diagnosis requests for diabetes."""
    if request.method == "GET":
        return render_template("conditions/diabetes.html")
    else:
        ORDER = [
            "pregnancies",
            "plasma",
            "diastolic",
            "skin",
            "insulin",
            "bmi",
            "pedigree",
            "age",
        ]
        observation = []
        for name in ORDER:
            observation.append(float(request.form.get(name)))
        model = DiagnosticModel()
        with open("./api/diabetes/model.json") as f:
            model.from_json(f.read())
        diagnosis, confidence_score = model.diagnose(observation)
        return render_template(
            "conditions/diagnosis.html",
            condition="diabetes",
            diagnosis=diagnosis,
            confidence_score=confidence_score,
        )


@app.route("/conditions/heart-disease", methods=["GET", "POST"])
def heart_disease():
    """Handle diagnosis requests for heart disease."""
    if request.method == "GET":
        return render_template("conditions/heart-disease.html")
    else:
        ORDER = [
            "age",
            "sex",
            "chest",
            "pressure",
            "serum",
            "sugar",
            "elec",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "vessels",
            "thal",
        ]
        observation = []
        for name in ORDER:
            observation.append(float(request.form.get(name)))
        model = DiagnosticModel()
        with open("./api/heart_disease/model.json") as f:
            model.from_json(f.read())
        diagnosis, confidence_score = model.diagnose(observation)
        return render_template(
            "conditions/diagnosis.html",
            condition="heart disease",
            diagnosis=diagnosis,
            confidence_score=confidence_score,
        )


@app.route("/docs/breast-cancer")
def breast_cancer_docs():
    """Return the documentation for breast cancer dataset."""
    return send_file("./static/docs/breast-cancer-docs.pdf")
