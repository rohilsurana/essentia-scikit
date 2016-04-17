import json, sys
from pprint import pprint
from os import listdir
from os.path import isfile, join
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GMM
from sklearn.cluster import KMeans
import numpy as np
from sklearn.svm import SVC
from tabulate import tabulate


main_path = sys.argv[1]
all_features = ['centroid', 'flatness', 'loudness', 'mfcc', 'roll_off', 'spread', 'skewness', 'kurtosis',
                'zero_crossing_rate']
all_genres = [join(main_path, "Carnatic/"), join(main_path, "Filmy/"), join(main_path, "Ghazal/"), join(main_path,
                                                                                                        "Hindustani/")]


def get_training_data(features, genres):
    files = {}
    X = []
    y = []
    for index, genre in enumerate(genres):
        for feature in features:
            path = join(genre, feature)
            files[feature] = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
            files[feature].sort()
        for x in range(0, 125):
            all_features_data = []
            for feature in features:
                with open(files[feature][x]) as data_file:
                    data = json.load(data_file)
                main_data = data["lowlevel"]
                try:
                    feature_data = main_data[feature][0]
                except KeyError:
                    feature_data = main_data[feature]["mean"]
                if isinstance(feature_data, list):
                    all_features_data.extend(feature_data)
                else:
                    all_features_data.append(feature_data)
            X.append(all_features_data)
            y.append(index)
    return X, y


def train(X, y):
    neigh = KNeighborsClassifier(n_neighbors=12)
    neigh.fit(X, y)

    means = KMeans(n_clusters=4)
    means.fit(X, y)

    mixture = GMM(n_components=4)
    mixture.fit(X, y)

    A = np.array(X)
    b = np.array(y)

    machine = SVC(decision_function_shape='ovr', kernel='rbf', gamma=0.00001)
    machine.fit(A, b)

    return neigh, means, mixture, machine


def test(features, genres, neigh, means, mixture, machine):
    for index, genre in enumerate(genres):
        T = []
        files = {}
        for feature in features:
            path = join(genre, feature)
            files[feature] = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
            files[feature].sort()
        for x in range(126, 151):
            all_features_data = []
            for feature in features:
                with open(files[feature][x]) as data_file:
                    data = json.load(data_file)
                main_data = data["lowlevel"]
                try:
                    feature_data = main_data[feature][0]
                except KeyError:
                    feature_data = main_data[feature]["mean"]
                if isinstance(feature_data, list):
                    all_features_data.extend(feature_data)
                else:
                    all_features_data.append(feature_data)
            T.append(all_features_data)
        knn = neigh.predict(T)
        gmm = mixture.predict(T)
        svm = machine.predict(T)
        kmeans = means.predict(T)
        table = [["kNN  ", sum(1 for x in knn.tolist() if x == index), knn],
                 ["kMeans  ", sum(1 for x in kmeans.tolist() if x == index), kmeans],
                 ["GMM  ", sum(1 for x in gmm.tolist() if x == index), gmm],
                 ["SVM  ", sum(1 for x in svm.tolist() if x == index), svm]
                 ]
        print "#####################", genre,"#####################"
        print tabulate(table)


def main():
    X, y = get_training_data(all_features, all_genres)
    neigh, means, mixture, machine = train(X, y)
    test(all_features, all_genres, neigh, means, mixture, machine)


if __name__ == '__main__':
    main()
