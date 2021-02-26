import csv
import pickle
import numpy as np


class DataProcessor:

    def __init__(self, path='rain_dataset/weatherAUS.csv'):

        # Read data from file
        with open(path) as file:
            csv_reader = csv.reader(file, delimiter=',')
            self.rawdata = np.array(list(csv_reader))
            file.close()

        # Extract relevant slices
        self.columns = self.rawdata[0, 1:]
        self.data = self.rawdata[1:, 1:]

        # Dictionaries for categorical data
        self.dirn = ['N', 'NNW', 'NW', 'WNW', 'W', 'WSW', 'SW', 'SSW', 'S', 'SSE', 'SE', 'ESE', 'E', 'ENE', 'NE', 'NNE']
        self.locn = list(set(self.data[:, 0]))
        self.pred = ['No', 'Yes']
        self.dirn = {d: i for i, d in enumerate(self.dirn)}
        self.locn = {d: i for i, d in enumerate(self.locn)}
        self.pred = {d: i for i, d in enumerate(self.pred)}

        # Initialize column stats
        self.min_values = np.zeros_like(self.columns)
        self.max_values = np.zeros_like(self.columns)
        self.mid_values = np.zeros_like(self.columns)

        # Process data
        self.quantify_data()
        self.compute_stats()
        self.impute_data()

        # Normalize data
        self.features = self.data[:, :-2].astype(float)
        self.labels = self.data[:, -2:].astype(float).astype(int)
        self.normalize_data()

    def quantify_data(self):

        # Build master dictionary
        translator = dict(self.dirn, **self.locn)
        translator.update(self.pred)

        # Remove non numeric data
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                self.data[i, j] = translator.get(self.data[i, j], self.data[i, j])

    def compute_stats(self):

        # Evaluate stats
        for j in range(self.data.shape[1]):
            col_data = [float(i) for i in self.data[:, j] if i != 'NA']
            self.min_values[j] = min(col_data)
            self.max_values[j] = max(col_data)
            self.mid_values[j] = sum(col_data) / len(col_data)

        # Convert to float
        self.min_values = self.min_values.astype(float)
        self.max_values = self.max_values.astype(float)
        self.mid_values = self.mid_values.astype(float)

    def impute_data(self):

        # Fill missing data
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                self.data[i, j] = self.mid_values[j] if self.data[i, j] == 'NA' else self.data[i, j]

    def normalize_data(self):

        # Normalize data
        for j in range(self.features.shape[1]):
            self.features[:, j] -= self.min_values[j]
            self.features[:, j] /= (self.max_values[j] - self.min_values[j])


if __name__ == '__main__':

    # Save the data object
    obj = DataProcessor()
    file = open('rain_dataset/processed_data', 'wb')
    pickle.dump(obj, file)
    file.close()
