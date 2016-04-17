import json, sys
from pprint import pprint
from os import listdir
from os.path import isfile, join

X = []
y = []

mypath = sys.argv[1]
features = ['centroid', 'flatness', 'loudness', 'mfcc', 'roll_off', 'spread_skewness_kurtosis', 'zero_crossing_rate']

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


# # For Filmy music range 1 to 125 files sorted alphabetically
# files = [join(filmy, f) for f in listdir(filmy) if isfile(join(filmy, f))]
# files.sort()

# for file in files[1:125]:
# 	with open(file) as data_file:
# 		data = json.load(data_file)
# 	main_data = data["lowlevel"]
# 	try:
# 		mfcc = main_data["mfcc"][0]
# 	except KeyError:
# 		mfcc = main_data["mfcc"]["mean"]
# 	X.append(mfcc)
# 	y.append(1)



# # For Ghazal music range 1 to 125 files sorted alphabetically
# files = [join(ghazal, f) for f in listdir(ghazal) if isfile(join(ghazal, f))]
# files.sort()

# for file in files[1:125]:
# 	with open(file) as data_file:
# 		data = json.load(data_file)
# 	main_data = data["lowlevel"]
# 	try:
# 		mfcc = main_data["mfcc"][0]
# 	except KeyError:
# 		mfcc = main_data["mfcc"]["mean"]
# 	X.append(mfcc)
# 	y.append(2)

# # For Hindustani music range 1 to 125 files sorted alphabetically
# files = [join(hindustani, f) for f in listdir(hindustani) if isfile(join(hindustani, f))]
# files.sort()

# for file in files[1:125]:
# 	with open(file) as data_file:
# 		data = json.load(data_file)
# 	main_data = data["lowlevel"]
# 	try:
# 		mfcc = main_data["mfcc"][0]
# 	except KeyError:
# 		mfcc = main_data["mfcc"]["mean"]
# 	X.append(mfcc)
# 	y.append(3)

from sklearn.externals import joblib
#############################################################
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)
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

clf = SVC(decision_function_shape='ovo', kernel='linear')
print(clf.fit(A, b))
# print(clf.predict(X))
#############################################################




for genre in genres:
	T = []
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
		T.append(all_features_data)
	print(genre)
	print("kNN")
	print(neigh.predict(T))
	print(sum(1 for x in neigh.predict(T).tolist() if x==0))
	print("kMeans")
	print(means.predict(T))
	print(sum(1 for x in means.predict(T).tolist() if x==0))
	print("GMM")
	print(mixture.predict(T))
	print(sum(1 for x in mixture.predict(T).tolist() if x==0))
	print("SVM")
	print(clf.predict(T))
	print(sum(1 for x in clf.predict(T).tolist() if x==0))
	print("#############################################################")	




# # For Carnatic music range 126 to 151 files sorted alphabetically as test cases
# files = [join(carnatic, f) for f in listdir(carnatic) if isfile(join(carnatic, f))]
# files.sort()
# T = []
# for file in files[126:151]:
# 	with open(file) as data_file:
# 		data = json.load(data_file)
# 	main_data = data["lowlevel"]
# 	try:
# 		mfcc = main_data["mfcc"][0]
# 	except KeyError:
# 		mfcc = main_data["mfcc"]["mean"]
# 	T.append(mfcc)

# print("Carnatic")
# print("kNN")
# # print(neigh.predict(T))
# print(sum(1 for x in neigh.predict(T).tolist() if x==0))
# print("kMeans")
# # print(means.predict(T))
# print(sum(1 for x in means.predict(T).tolist() if x==0))
# print("GMM")
# # print(mixture.predict(T))
# print(sum(1 for x in mixture.predict(T).tolist() if x==0))
# print("SVM")
# # print(clf.predict(T))
# print(sum(1 for x in clf.predict(T).tolist() if x==0))
# print("#############################################################")

# files = [join(filmy, f) for f in listdir(filmy) if isfile(join(filmy, f))]
# files.sort()
# T = []
# for file in files[126:151]:
# 	with open(file) as data_file:
# 		data = json.load(data_file)
# 	main_data = data["lowlevel"]
# 	try:
# 		mfcc = main_data["mfcc"][0]
# 	except KeyError:
# 		mfcc = main_data["mfcc"]["mean"]
# 	T.append(mfcc)


# print("Filmy")
# print("kNN")
# # print(neigh.predict(T))
# print(sum(1 for x in neigh.predict(T).tolist() if x==1))
# print("kMeans")
# # print(means.predict(T))
# print(sum(1 for x in means.predict(T).tolist() if x==1))
# print("GMM")
# # print(mixture.predict(T))
# print(sum(1 for x in mixture.predict(T).tolist() if x==1))
# print("SVM")
# # print(clf.predict(T))
# print(sum(1 for x in clf.predict(T).tolist() if x==1))
# print("#############################################################")

# files = [join(ghazal, f) for f in listdir(ghazal) if isfile(join(ghazal, f))]
# files.sort()
# T = []
# for file in files[126:151]:
# 	with open(file) as data_file:
# 		data = json.load(data_file)
# 	main_data = data["lowlevel"]
# 	try:
# 		mfcc = main_data["mfcc"][0]
# 	except KeyError:
# 		mfcc = main_data["mfcc"]["mean"]
# 	T.append(mfcc)


# print("Ghazal")
# print("kNN")
# # print(neigh.predict(T))
# print(sum(1 for x in neigh.predict(T).tolist() if x==2))
# print("kMeans")
# # print(means.predict(T))
# print(sum(1 for x in means.predict(T).tolist() if x==2))
# print("GMM")
# # print(mixture.predict(T))
# print(sum(1 for x in mixture.predict(T).tolist() if x==2))
# print("SVM")
# # print(clf.predict(T))
# print(sum(1 for x in clf.predict(T).tolist() if x==2))
# print("#############################################################")

# files = [join(hindustani, f) for f in listdir(hindustani) if isfile(join(hindustani, f))]
# files.sort()
# T = []
# for file in files[126:151]:
# 	with open(file) as data_file:
# 		data = json.load(data_file)
# 	main_data = data["lowlevel"]
# 	try:
# 		mfcc = main_data["mfcc"][0]
# 	except KeyError:
# 		mfcc = main_data["mfcc"]["mean"]
# 	T.append(mfcc)


# print("Hindustani")
# print("kNN")
# # print(neigh.predict(T))
# print(sum(1 for x in neigh.predict(T).tolist() if x==3))
# print("kMeans")
# # print(means.predict(T))
# print(sum(1 for x in means.predict(T).tolist() if x==3))
# print("GMM")
# # print(mixture.predict(T))
# print(sum(1 for x in mixture.predict(T).tolist() if x==3))
# print("SVM")
# # print(clf.predict(T))
# print(sum(1 for x in clf.predict(T).tolist() if x==3))