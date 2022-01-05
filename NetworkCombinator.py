import os
import csv
import numpy as np
from sklearn.metrics import confusion_matrix

class NetworkCombinator:
    def __init__(self, 
                result_csv_file_path : str,
                netout_csv_file_path: str) -> None:
        self._result_csv_file_path = result_csv_file_path

        self._turn_on_googleNet = 1
        self._turn_on_resNet50 = 1
        self._turn_on_resNet101 = 1
        self._turn_on_denseNet201 = 1

        self._googleNet_acc = 1
        self._resNet50_acc = 1
        self._resNet101_acc = 1
        self._denseNet201_acc = 1
        
        self._netout_csv_file_path = netout_csv_file_path


    def read_data(self):
        file = open(self._result_csv_file_path)
        csvreader = csv.reader(file)
        header = next(csvreader)
        
        rows = []
        for row in csvreader:
            rows.append(row)
        file.close()

        self._create_dictionaries(header, rows)

        file = open(self._netout_csv_file_path)
        csvreader = csv.reader(file)
        self.rows = []
        for row in csvreader:
            self.rows.append(row)
        file.close()


    def _create_dictionaries(self, header, rows):
        self.googleNet_dict = {header[1] : float(rows[0][1]) * self._turn_on_googleNet * self._googleNet_acc}
        self.ResNet50_dict = {header[1] : float(rows[1][1]) * self._turn_on_resNet50 * self._resNet50_acc}
        self.ResNet101_dict = {header[1] : float(rows[2][1]) * self._turn_on_resNet101 * self._resNet101_acc}
        self.DenseNet201_dict = {header[1] : float(rows[3][1]) * self._turn_on_denseNet201 * self._denseNet201_acc}

        for i in range(2, len(header)):
            self.googleNet_dict[header[i]] = float(rows[0][i]) * self._turn_on_googleNet * self._googleNet_acc
            self.ResNet50_dict[header[i]] = float(rows[1][i]) * self._turn_on_resNet50 * self._resNet50_acc
            self.ResNet101_dict[header[i]] = float(rows[2][i]) * self._turn_on_resNet101 * self._resNet101_acc
            self.DenseNet201_dict[header[i]] = float(rows[3][i]) * self._turn_on_denseNet201 * self._denseNet201_acc


    def select_networks(self, include_googleNet, include_resNet50, include_resNet101, include_denseNet201):
        self._turn_on_googleNet = include_googleNet
        self._turn_on_resNet50 = include_resNet50
        self._turn_on_resNet101 = include_resNet101
        self._turn_on_denseNet201 = include_denseNet201

    def set_networks_accuracy(self, googleNet_acc, resNet50_acc, resNet101_acc, denseNet201_acc):
        self._googleNet_acc = googleNet_acc
        self._resNet50_acc = resNet50_acc
        self._resNet101_acc = resNet101_acc
        self._denseNet201_acc = denseNet201_acc


    def combine_networks(self):
        correct_class = []
        detect_class = []
        for i in range(0, len(self.rows)):
            labels = list(set(self.rows[i][1:]))
            weight = np.zeros(len(labels))
            for j in range(1, 5):
                for k in range(0, len(labels)):
                    if labels[k] == self.rows[i][j]:
                        if j==1:
                            weight[k] = weight[k] + self.googleNet_dict[labels[k]]
                        elif j==2:
                            weight[k] = weight[k] + self.ResNet50_dict[labels[k]]
                        elif j==3:
                            weight[k] = weight[k] + self.ResNet101_dict[labels[k]]
                        else:
                            weight[k] = weight[k] + self.DenseNet201_dict[labels[k]]
                
            weight = weight / np.sum(weight)
            pos = np.argmax(weight)
            detect_class.append(labels[pos])
            correct_class.append(self.rows[i][0])

        cf = confusion_matrix(correct_class, detect_class)
        accuracy = np.trace(cf)/np.sum(cf)*100

        return cf, accuracy


folder_path = os.path.dirname(os.path.realpath(__file__))
result_csv_file_path = f'{folder_path}/results_21.csv'
net_out_csv_file_path = f'{folder_path}/21 classes validation.csv'

net = NetworkCombinator(result_csv_file_path, net_out_csv_file_path)
net.select_networks(1,1,1,1)
net.set_networks_accuracy(0.485, 0.55, 0.497, 0.664)
net.read_data()
cf, acc = net.combine_networks()

print(acc)