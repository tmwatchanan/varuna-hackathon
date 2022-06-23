# varuna-hackathon

- Team Name: **ktaff**
- Date: 2022-06-19
- `ktaff-final_submission.csv` is the predicted output on the test set.

## Steps
1. `Sentinel_2.ipynb` is used to process satelite images from Sentinel-2. We can get 348 features extracted from multiple bands. It was run two times (which requires manually change of the cells, the latest version committed to this repo is to generate the second one):
    1. To generate EVI, NDWI, SAVI, and Band1 --> `train_max_evi_ndwi_savi_b1.csv` and `test_max_evi_ndwi_savi_b1.csv`
    2. To generate NDVI, Misra Yellow Vegetation Index, and Band5 --> `train_NDVI_misra_b5.csv` and `test_NDVI_misra_b5.csv`
2. Run `SCL_Train.ipynb` or `SCL_Test.ipynb` to process SCL data on the training dataset or the test dataset. They also combine the specific SCL classes with the extracted features (from Step 1). They create `train_set.csv` and `test_set.csv`.
4. Run `Classification.ipynb` which utilizes the ensemble method of 5 different classifiers. It produces `ktaff-final_submission.csv` as the final output.
