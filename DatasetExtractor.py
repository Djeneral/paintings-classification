import os
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize

class DatasetExtractor:
    def __init__(self, 
                folder_path : str,
                csv_file_path : int) -> None:
        self._folder_path = folder_path
        self._csv_file_path = csv_file_path

        self._header, self._rows = self._read_csv_file()
        self.genres = self._extract_genres()


    def _read_csv_file(self):
        file = open(self._csv_file_path)
        csvreader = csv.reader(file)
        header = next(csvreader)
        
        rows = []
        for row in csvreader:
            rows.append(row)
        file.close()

        return header, rows


    def _extract_genres(self):
        genres = []
        for i in range(0, len(self._rows)):
            genres.append(self._rows[i][3])

        genres = sorted(list(set(genres)))

        return genres


    def split_dataset(self, train_ratio:float, val_ratio:float, test_ratio:float) -> None:
        self._create_dir()

        train_idx = 0
        val_idx = 0
        test_idx = 0

        for i in tqdm (range (0, len(self._rows)), desc="Dataset spliting..."):
            artist_name = self._rows[i][1].replace(" ", "_")
            path = f'{self._folder_path}/images/images/{artist_name}/'
            paintings = int(self._rows[i][6])
            image_paths = os.listdir(path)
            genre = self._rows[i][3]

            train_path = f'{self._folder_path}/dataset/train'
            val_path = f'{self._folder_path}/dataset/validation'
            test_path = f'{self._folder_path}/dataset/test'
            
            for j, img_path in enumerate(image_paths):
                img = plt.imread(f'{path}/{img_path}')
                img = resize(img, [224, 224])
                if j/paintings > train_ratio + val_ratio:
                    plt.imsave(f'{test_path}/{genre}/{genre}_{test_idx}.jpg', img)
                    test_idx = test_idx + 1
                elif j/paintings > train_ratio:
                    plt.imsave(f'{val_path}/{genre}/{genre}_{val_idx}.jpg', img)
                    val_idx = val_idx + 1
                else:
                    plt.imsave(f'{train_path}/{genre}/{genre}_{train_idx}.jpg', img)
                    train_idx = train_idx + 1
        
        print(test_idx, val_idx, train_idx)

           
    def _create_dir(self):
        if not os.path.isdir(f'{self._folder_path}/dataset'):
            os.mkdir(f'{self._folder_path}/dataset')
            os.mkdir(f'{self._folder_path}/dataset/train')
            os.mkdir(f'{self._folder_path}/dataset/test')
            os.mkdir(f'{self._folder_path}/dataset/validation')

        for i in range(0, len(self._rows)):
            genre = self._rows[i][3]
            path_train = f'{self._folder_path}/dataset/train/{genre}'
            path_val = f'{self._folder_path}/dataset/validation/{genre}'
            path_test = f'{self._folder_path}/dataset/test/{genre}'
            if not os.path.isdir(path_train):
                os.mkdir(path_train)
                os.mkdir(path_val)
                os.mkdir(path_test)


folder_path = os.path.dirname(os.path.realpath(__file__))
csv_file_path = f'{folder_path}/genres.csv'

dataset_extractor = DatasetExtractor(folder_path, csv_file_path)
dataset_extractor.split_dataset(0.7, 0.15, 0.15)
    