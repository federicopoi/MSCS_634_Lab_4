# MSCS 634 — Lab 4

Federico Poi · MSCS-634-M50 Advanced Big Data and Data Mining

## Purpose
I compare simple, multiple, and polynomial regression on the scikit-learn Diabetes dataset, then use Ridge and Lasso to see where regularization improves the fit. The notebook includes exploration, all four required error metrics, prediction plots, and an analysis of overfitting.

## Data and evaluation
The data contains 442 patients, ten baseline measurements, and a continuous disease-progression score measured one year later. There are no missing values, duplicate predictor rows, or nonfinite values, so no records were removed or imputed.

BMI is the single predictor for simple regression; the other models use all ten features. An 80/20 split with `random_state=42` gives 353 training and 89 test records. Five shuffled training folds select alpha and compare models. I load the unscaled data and fit both scaling and polynomial transformations inside pipelines. Test scores never determine the feature choice, degree, or alpha.

## Results
All values below come from the same held-out test set. Lower MAE, MSE, and RMSE are better; higher R² is better.

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Mean baseline | 64.01 | 5361.53 | 73.22 | -0.0120 |
| Simple linear (BMI) | 52.26 | 4061.83 | 63.73 | 0.2334 |
| Multiple linear | 42.79 | 2900.19 | 53.85 | 0.4526 |
| Polynomial degree 2 | 43.58 | 3096.03 | 55.64 | 0.4156 |
| Polynomial degree 3 | 164.85 | 82446.05 | 287.13 | -14.5613 |
| Ridge | 42.81 | 2892.01 | 53.78 | 0.4541 |
| Lasso | 42.81 | 2884.68 | 53.71 | 0.4555 |
| Ridge + quadratic | 41.47 | 2649.61 | 51.47 | 0.4999 |
| Lasso + quadratic | 42.50 | 2685.62 | 51.82 | 0.4931 |

The other measurements help: multiple regression reduces test RMSE by about ten points compared with BMI alone. Unpenalized polynomial features go the other way. The cubic model lowers training RMSE to 27.31 but raises test RMSE to 287.13. It fits training patterns that do not carry over, and the large, correlated expansion is unstable.

Regularization has a modest effect on the original ten features and a clearer effect on the quadratic expansion. Quadratic Ridge has the lowest observed test RMSE, 51.47, using alpha 100. Quadratic Lasso is close at 51.82, using alpha 4.6416, while retaining 11 of 65 coefficients. On the original features, the selected Lasso alpha of 0.1 retains all ten coefficients. Lasso can remove terms, but does not always do so at the chosen penalty.

Training cross-validation selects Lasso on the original features (CV RMSE 55.40), rather than quadratic Ridge (56.89). I keep that selection separate from the observed test ranking. The linear models have nearly identical validation scores; one split is not enough to establish a reliable winner.

## Decisions and challenges
- Scaling is fitted within each fold, including a second scaler after polynomial expansion. This keeps penalties comparable across generated terms and avoids preprocessing leakage.
- The 65 quadratic and 285 cubic columns include redundant terms, particularly powers of the two-valued sex feature. Their count is not the number of independent dimensions.
- Very small polynomial Lasso penalties produced convergence warnings in an initial check. The final grid starts at 0.1, uses up to 100,000 iterations, and treats convergence warnings as errors. All final fits completed without them. Alpha values are only best within the tested grids.
- Larger alpha values eventually underfit; the alpha and sparsity plots show this tradeoff. Ridge and Lasso use different objective scalings, so their alpha values are not directly comparable.

These results concern prediction on a small dataset. Correlations and fitted coefficients do not establish causes of disease progression.

## Run the notebook
Use Python 3.11 or newer (tested with Python 3.14.3 and scikit-learn 1.8.0):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m notebook MSCS_634_Lab_4.ipynb
```

Choose **Restart Kernel and Run All Cells**. No dataset download is needed. The saved notebook already contains the executed outputs.

## Files
- `MSCS_634_Lab_4.ipynb`: completed notebook with code, outputs, plots, and analysis.
- `screenshots/`: eleven exported figures.
- `results/`: comparison metrics, polynomial-degree results, selected alphas, and run versions.
- `requirements.txt`: packages for rerunning the lab.

## References
- [scikit-learn Diabetes dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
- [Linear models and regularization](https://scikit-learn.org/stable/modules/linear_model.html)
- [Data leakage and pipelines](https://scikit-learn.org/stable/common_pitfalls.html)
