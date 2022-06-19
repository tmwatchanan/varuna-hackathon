# varuna-hackathon

Team Name: **ktaff**
Date: 2022-06-19

## Steps
1. `Sentinel_2.ipynb` is used to process satelite images from Sentinel-2. We can get 348 features extracted from multiple bands.
2. Run `SCL_Train.ipynb` or `SCL_Test.ipynb` to process SCL data on the training dataset or the test dataset.
3. Combine the specific SCL classes (from Step 2) with the extracted features (from Step 1)
4. Run `Classification.ipynb` which utilizes the ensemble method of 5 different classifiers.

- `ktaff-final_submission.csv` is the predicted output on the test set.