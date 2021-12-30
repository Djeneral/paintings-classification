# paintings-classification

## Database
The database was downloaded from the Kaggle website and can be found at this [link](https://www.kaggle.com/ikarus777/best-artworks-of-all-time). As the original database provides information about the painting era to which the painter belongs, we decided to create a system for recognizing artistic style.  

The artists.csv file (from original dataset) contains several genres for some artists, which cannot be properly classified, so only one genre per artist is set, so we have 21 classes. You can download the reduced file genere_21.csv from [here](https://drive.google.com/file/d/14OW_zfs2XDyGPyiTTCNbT_BKv18obw_v/view?usp=sharing
). Also, some classes have few examples and the operation of the system was examined for the 11 most common classes. You can download the reduced file genres_11.csv [here](https://drive.google.com/file/d/142w6ZCeTfe_k9Q-7pfv8q5_cFNuCsqxa/view?usp=sharing).

In order to create a database that the neural network would understand, it is necessary to divide the data into training, test and validation. The division into subsets is implemented using the DatasetExtractor class in the DatasetExtractor.py class.

### DatasetExtractor
Input arguments:
Image folder and database destriptor in csv file for class constructor. Train ratio, validation ratio and test ratio for split dataset function.
Example:
```
dataset_extractor = DatasetExtractor(images_folder_path, csv_file_path)
dataset_extractor.split_dataset(train_ratio = 0.7, val_ratio = 0.15, test_ratio = 0.15)
```