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

## Neural network models
For the classifier, pre-trained convolutional networks are used, which are adapted for classification into 11 and 21 classes, respectively. The networsk was trained in MatLab Deep Learning Toolbox. Neural networks vgg16, vgg19, squeezenet and resnet18 are not used because they are less complex than the GoogleNet network which has shown significantly worse results. More complex architectures were not used due to the hardware configuration of the available device

### GoogleNet
GoogleNet architecture and training setting are implemented in config_googleNet.m file. The number of classes and the validation set are passed as function parameters. 

Example:
```
[lgraph, options] = config_googleNet(number_of_classes, validation_imds);
```

### ResNet50
ResNet50 architecture and training setting are implemented in config_resnet50.m file. The number of classes and the validation set are passed as function parameters.  

Example:
```
[lgraph, options] = config_resnet50(number_of_classes, validation_imds);
```

### ResNet101
ResNet101 architecture and training setting are implemented in config_resnet101.m file. The number of classes and the validation set are passed as function parameters.  

Example:
```
[lgraph, options] = config_resnet101(number_of_classes, validation_imds);
```

## Results
Results for GoogleNet you can see [here](https://drive.google.com/drive/folders/1DnWrwS7fTQPFDJ3YzBPYe82oTdysp-Y1?usp=sharing).

Results for ResNet50 you can see [here](https://drive.google.com/drive/folders/1MA3GT-hBS6X_8dl0Wb39DeMGDJF3qyvx?usp=sharing).

Results for ResNet101 you can see [here](https://drive.google.com/drive/folders/1-HS6x6vj2O_3BqYOQz-ZQNCnMjtqRGSx?usp=sharing).


|                    |            | 11 classes |      |            | 21 classes |      |
|:------------------:|:----------:|:----------:|:----:|:----------:|:----------:|:----:|
|**Network**         | Train      | Validation | Test | Train      | Validation | Test |
| GoogleNet!         | 78.0%      | 53.8%      | 51.1%| 81.6%      | 42.0%      | 41.0%|
| ResNet50           | 95.6%      | 60.5%      | 59.6%| 92.8%      | 55.0%      | 52.2%|
| ResNet101          | 83.5%      | 53.1%      | 47.2%| ??.?%      | ??.?%      | ??.?%|