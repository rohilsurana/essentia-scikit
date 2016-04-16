import json, sys
from pprint import pprint
from os import listdir
from os.path import isfile, join

X = []
y = []

mypath = sys.argv[1]

carnatic = join(mypath, "Carnatic/mfcc")
filmy = join(mypath, "Filmy/mfcc")
ghazal = join(mypath, "Ghazal/mfcc")
hindustani = join(mypath, "Hindustani/mfcc")

# For Carnatic music range 1 to 125 files sorted alphabetically
files = [join(carnatic, f) for f in listdir(carnatic) if isfile(join(carnatic, f))]
files.sort()

for file in files[1:125]:
	with open(file) as data_file:
		data = json.load(data_file)
	main_data = data["lowlevel"]
	try:
		mfcc = main_data["mfcc"][0]
	except KeyError:
		mfcc = main_data["mfcc"]["mean"]
	X.append(mfcc)
	y.append(0)


# For Filmy music range 1 to 125 files sorted alphabetically
files = [join(filmy, f) for f in listdir(filmy) if isfile(join(filmy, f))]
files.sort()

for file in files[1:125]:
	with open(file) as data_file:
		data = json.load(data_file)
	main_data = data["lowlevel"]
	try:
		mfcc = main_data["mfcc"][0]
	except KeyError:
		mfcc = main_data["mfcc"]["mean"]
	X.append(mfcc)
	y.append(1)



# For Ghazal music range 1 to 125 files sorted alphabetically
files = [join(ghazal, f) for f in listdir(ghazal) if isfile(join(ghazal, f))]
files.sort()

for file in files[1:125]:
	with open(file) as data_file:
		data = json.load(data_file)
	main_data = data["lowlevel"]
	try:
		mfcc = main_data["mfcc"][0]
	except KeyError:
		mfcc = main_data["mfcc"]["mean"]
	X.append(mfcc)
	y.append(2)

# For Hindustani music range 1 to 125 files sorted alphabetically
files = [join(hindustani, f) for f in listdir(hindustani) if isfile(join(hindustani, f))]
files.sort()

for file in files[1:125]:
	with open(file) as data_file:
		data = json.load(data_file)
	main_data = data["lowlevel"]
	try:
		mfcc = main_data["mfcc"][0]
	except KeyError:
		mfcc = main_data["mfcc"]["mean"]
	X.append(mfcc)
	y.append(3)


#############################################################
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X,y)
# print(neigh.predict(X))
#############################################################
from sklearn.cluster import KMeans

means = KMeans(n_clusters=4)
means.fit(X,y)
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

clf = SVC()
clf.fit(A, b)
# print(clf.predict(X))
#############################################################



# For Carnatic music range 126 to 150 files sorted alphabetically as test cases
files = [join(carnatic, f) for f in listdir(carnatic) if isfile(join(carnatic, f))]
files.sort()
T = []
for file in files[126:150]:
	with open(file) as data_file:
		data = json.load(data_file)
	main_data = data["lowlevel"]
	try:
		mfcc = main_data["mfcc"][0]
	except KeyError:
		mfcc = main_data["mfcc"]["mean"]
	T.append(mfcc)

print("Carnatic")
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

files = [join(filmy, f) for f in listdir(filmy) if isfile(join(filmy, f))]
files.sort()
T = []
for file in files[126:150]:
	with open(file) as data_file:
		data = json.load(data_file)
	main_data = data["lowlevel"]
	try:
		mfcc = main_data["mfcc"][0]
	except KeyError:
		mfcc = main_data["mfcc"]["mean"]
	T.append(mfcc)


print("Filmy")
print("kNN")
print(neigh.predict(T))
print(sum(1 for x in neigh.predict(T).tolist() if x==1))
print("kMeans")
print(means.predict(T))
print(sum(1 for x in means.predict(T).tolist() if x==1))
print("GMM")
print(mixture.predict(T))
print(sum(1 for x in mixture.predict(T).tolist() if x==1))
print("SVM")
print(clf.predict(T))
print(sum(1 for x in clf.predict(T).tolist() if x==1))
print("#############################################################")

files = [join(ghazal, f) for f in listdir(ghazal) if isfile(join(ghazal, f))]
files.sort()
T = []
for file in files[126:150]:
	with open(file) as data_file:
		data = json.load(data_file)
	main_data = data["lowlevel"]
	try:
		mfcc = main_data["mfcc"][0]
	except KeyError:
		mfcc = main_data["mfcc"]["mean"]
	T.append(mfcc)


print("Ghazal")
print("kNN")
print(neigh.predict(T))
print(sum(1 for x in neigh.predict(T).tolist() if x==2))
print("kMeans")
print(means.predict(T))
print(sum(1 for x in means.predict(T).tolist() if x==2))
print("GMM")
print(mixture.predict(T))
print(sum(1 for x in mixture.predict(T).tolist() if x==2))
print("SVM")
print(clf.predict(T))
print(sum(1 for x in clf.predict(T).tolist() if x==2))
print("#############################################################")

files = [join(hindustani, f) for f in listdir(hindustani) if isfile(join(hindustani, f))]
files.sort()
T = []
for file in files[126:150]:
	with open(file) as data_file:
		data = json.load(data_file)
	main_data = data["lowlevel"]
	try:
		mfcc = main_data["mfcc"][0]
	except KeyError:
		mfcc = main_data["mfcc"]["mean"]
	T.append(mfcc)


print("Hindustani")
print("kNN")
print(neigh.predict(T))
print(sum(1 for x in neigh.predict(T).tolist() if x==3))
print("kMeans")
print(means.predict(T))
print(sum(1 for x in means.predict(T).tolist() if x==3))
print("GMM")
print(mixture.predict(T))
print(sum(1 for x in mixture.predict(T).tolist() if x==3))
print("SVM")
print(clf.predict(T))
print(sum(1 for x in clf.predict(T).tolist() if x==3))