Model
=====
- Type: Balanced Random Forest

Dataset Source
==============
- The model was fitted on the following dataset:
  https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data

Dataset Predictors
==================
- The dataset contains the following 13 predictors that are used to predict heart disease:
    1) Age in years.
    2) Sex. 1 = male, 0 = female.
    3) Chest pain type: 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic
    4) Resting blood pressure in mmHg on admission to hospital.
    5) Serum cholesterol in mg/dl.
    6) Fasting blood sugar > 120 mg/dl? 1 = true, 0 = false.
    7. Resting electrocardiographic results.
       0 = normal
       1 = having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05mV)
       2 = showing probable or definite left ventricular hypertrophy by Estes' criteria
    8) thalach: Maximum heart rate achieved.
    9) exang: Exercised induced angina: 1 = yes, 0 = no
    10) oldpeak: ST depression induced by exercise relative to rest
    11) Slope of the peak exercise ST segment
        1 = upsloping
        2 = flat
        3 = downsloping
    12) Number of major blood vessels (0-3) colored by flourosopy.
    13) thal: 3 = normal, 6 = fixed defect, 7 = reversable defect
- The last i.e. 14th variable is the label itself.
    14) Number of major blood vessels with >50% diameter narrowing. Ranges from 0 to 4.
        For diagnosis, a single major blood vessel with >50% narrowing represents heart disease.
        i.e. Anything from 1 to 4 is heart disease, with 0 being a lack of heart disease.

Dataset Contributors
====================
- Stethobot would like to recognize the individuals and organizations responsible for this dataset.
- Thanks to the 4 authors who generously donated this dataset:
    1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
    2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
    3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
    4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.
- Thanks to UC Irvine for hosting this dataset within their repository:
    - Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
