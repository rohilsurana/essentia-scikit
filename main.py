import json, sys
from pprint import pprint
from os import listdir
from os.path import isfile, join

X = []
y = []

mypath = sys.argv[1]
features = ['centroid', 'flatness', 'loudness', 'mfcc', 'roll_off', 'spread_skewness_kurtosis', 'zero_crossing_rate']
features = ['mfcc']
genres = [join(mypath, "Carnatic/"), join(mypath, "Filmy/"), join(mypath, "Ghazal/"), join(mypath, "Hindustani/")]
files = {}

for index, genre in enumerate(genres):
	for feature in features:
		path = join(genre, feature)
		files[feature] = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
		files[feature].sort()
	for x in range(0,125):
		all_features_data = []
		for feature in features:
			with open(files[feature][x]) as data_file:
				data = json.load(data_file)
			main_data = data["lowlevel"]
			if feature =='spread_skewness_kurtosis':
				try:
					spread = main_data['spread'][0]
				except KeyError:
					spread = main_data['spread']["mean"]
				try:
					skewness = main_data['skewness'][0]
				except KeyError:
					skewness = main_data['skewness']["mean"]
				try:
					kurtosis = main_data["kurtosis"][0]
				except KeyError:
					kurtosis = main_data["kurtosis"]["mean"]
				all_features_data.append(spread)
				all_features_data.append(skewness)
				all_features_data.append(kurtosis)
			else:
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


from sklearn.externals import joblib
#############################################################
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=12)
neigh.fit(X,y)
# joblib.dump(neigh, 'KNN.mdl')
# print(neigh.predict(X))
#############################################################
from sklearn.cluster import KMeans

means = KMeans(n_clusters=4)
means.fit(X,y)
# joblib
# print(means.predict(X))
#############################################################
from sklearn.mixture import GMM

mixture = GMM(n_components=4)
mixture.fit(X,y)
# print(mixture.predict(X))
#############################################################
import numpy as np
from sklearn.svm import SVC

A = np.array(X)
b = np.array(y)

clf = SVC(decision_function_shape='ovo', kernel='poly')
clf.fit(A, b)
# print(clf.predict(X))
#############################################################




for index, genre in enumerate(genres):
	T = []
	for feature in features:
		path = join(genre, feature)
		files[feature] = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
		files[feature].sort()
	for x in range(126,151):
		all_features_data = []
		for feature in features:
			with open(files[feature][x]) as data_file:
				data = json.load(data_file)
			main_data = data["lowlevel"]
			if feature =='spread_skewness_kurtosis':
				try:
					spread = main_data['spread'][0]
				except KeyError:
					spread = main_data['spread']["mean"]
				try:
					skewness = main_data['skewness'][0]
				except KeyError:
					skewness = main_data['skewness']["mean"]
				try:
					kurtosis = main_data["kurtosis"][0]
				except KeyError:
					kurtosis = main_data["kurtosis"]["mean"]
				all_features_data.append(spread)
				all_features_data.append(skewness)
				all_features_data.append(kurtosis)
			else:
				try:
					feature_data = main_data[feature][0]
				except KeyError:
					feature_data = main_data[feature]["mean"]
				if isinstance(feature_data, list):
					all_features_data.extend(feature_data)
				else:
					all_features_data.append(feature_data)
		T.append(all_features_data)
	print("kNN")
	print(neigh.predict(T))
	print(sum(1 for x in neigh.predict(T).tolist() if x==index))
	print("kMeans")
	print(means.predict(T))
	print(sum(1 for x in means.predict(T).tolist() if x==index))
	print("GMM")
	print(mixture.predict(T))
	print(sum(1 for x in mixture.predict(T).tolist() if x==index))
	print("SVM")
	print(clf.predict(T))
	print(sum(1 for x in clf.predict(T).tolist() if x==index))
	print("#############################################################")	
