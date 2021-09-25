Model
=====
- Type: Balanced Random Forest

Dataset Source
==============
- The model was fit on the following dataset:
  https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data

Dataset Predictors
==================
- The first 2 columns are an ID number and the label: M for malignant, B for benign.
- The dataset's predictors are based off of a digitized image of a fine needle aspirate (FNA) of a breast mass.
- The dataset contains 30 predictors:
    - The first 10 are means of the following variables:
        1) Radius (mean of distances from center to points on the perimeter)
        2) Texture (standard deviation of gray-scale values)
        3) Perimeter
        4) Area
        5) Smoothness (local variation in radius lengths)
        6) Compactness (perimeter^2 / (area - 1))
        7) Concavity (severity of concave portions of the contour)
        8) Concave points (number of concave portions of the contour)
        9) Symmetry
        10) Fractal dimension ("coastline approximation" - 1)
    - The next 10 are standard errors of the first 10 variables.
    - The last 10 are the maximum of the first 10 variables.

Dataset Contributors
====================
- Stethobot would like to recognize the individuals and organizations responsible for this dataset.
- Thanks to the 3 authors who created this dataset:
    1. Dr. William H. Wolberg, General Surgery Dept.
       University of Wisconsin, Clinical Sciences Center
       Madison, WI 53792
       wolberg '@' eagle.surgery.wisc.edu
    2. W. Nick Street, Computer Sciences Dept.
       University of Wisconsin, 1210 West Dayton St., Madison, WI 53706
       street '@' cs.wisc.edu 608-262-6619
    3. Olvi L. Mangasarian, Computer Sciences Dept.
       University of Wisconsin, 1210 West Dayton St., Madison, WI 53706
       olvi '@' cs.wisc.edu
- Dataset was donated by the second author, Nick W. Street.
- Thanks to UC Irvine for hosting this dataset within their repository:
  Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
