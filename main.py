import math
import random
import numpy as np
import matplotlib as mpl
from PIL import Image
from scipy import fft
from scipy.spatial import distance
from matplotlib import pyplot as plt
from scipy.fftpack import dct

def open(name):
	img = Image.open(name)
	t = img.copy()
	img.close()
	return t

def toList(image):
	return list(image.getdata())

def toArray(matrix):
	vector = []

	for i in range(len(matrix)):
		for y in range(len(matrix[0])):
			vector.append(matrix[i][y])

	return vector

def human(number):
	human = [0] * 10
	for j in range(10):
		human[j] = open("./s" + str(number+1) + "/" + str(j+1) + ".pgm")
	return human

def convert(image):
	pixels = list(image.getdata())
	width, height = image.size
	pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
	return pixels

def Human():

	human = []

	for h in range(1,41):
		person = []
		for j in range(1,11):
			person.append(open("./s" + str(h) + "/" + str(j) + ".pgm"))
		human.append(person)

	return human


def distanceBetween(vec1, vec2):
	return distance.euclidean(vec1, vec2)

# Functions for histogram

def toHistogramVector(human):
	vectors = []
	for item in range(len(human)):
		vectors.append(human[item].histogram())
	return vectors

def compression(histogram, size):
	lenght = len(histogram)

	quantity = math.trunc(lenght/size)

	list = []

	start = 0

	for item in range(size-1):
		list.append(sum(histogram[start:start+quantity]))
		start=start + quantity
	list.append(sum(histogram[quantity*(size-1):lenght-1]))

	return list

def toHistogramVectorDynamic(human, size):
	vectors = []
	human = toHistogramVector(human)
	for item in range(len(human)):
		vectors.append(compression(human[item],size))
	return vectors

# / Functions for histogram

# Functions for Points

# Function for get Random coordinates of 200 points
def getRandomPoints():

	points = []
	for point in range(200):
		x = random.randint(0,111)
		y = random.randint(0,91)
		points.append([x,y])

	return points

# Get specially points of image
def getPointsFromHuman(human, coordinates):
	vector = []

	for point in coordinates:
		vector.append(human[point[0]][point[1]])

	return vector

def toRandomDynamic(human, quantity):
	vectors = []

	for person in human:
		vectors.append(getPointsFromHuman(convert(person), points[0:quantity]))

	return vectors
#/ Functions for Points

# Functions for Scale
def toScaleVector(human):
	vecHuman = []

	for hum in human:
		vecHuman.append(toList(hum.resize([69, 84])))   #75%

	return vecHuman

def toScaleVectorDynamic(human, size):
	vecHuman = []

	h = math.trunc(92*size)
	w = math.trunc(122*size)

	for hum in human:
		vecHuman.append(toList(hum.resize([h,w])))

	return vecHuman

#/ Functions for Scale

# Functions for DCT

def toDCTvectorDynamic(human, n):
	vecHuman = []

	for hum in human:
		vecHuman.append(dct(toList(hum.resize([n,n])), axis=0))

	return vecHuman

#/ Functions for DCT

# Functions for DFT
def toDFTvectorDynamic(human, n):
	vecHuman = []


	for hum in human:
		vecHuman.append(fft(toList(hum.resize([n,n])), axis=0))

	return vecHuman

#/ Functions for DFT

# Functions for Grad

def toGradVectorDynamicList(human, n):
	vecHuman = []

	for hum in human:
		vecHuman.append(np.gradient(toList(hum.resize([n,n]))))

	return vecHuman

#/ Functions for Grad

def Persent(arrFalse, arrTrue):
	result = []

	q = len(arrFalse)

	for index in range(q):
		if (arrFalse[index] < arrTrue[index]):
			result.append(index)

	return (q-len(result))/q*100

def MainData(vectors, quantity, klass):

	minTrue = []
	minFalse = []

	for b in range(10-quantity):
		minTrue.append(distanceBetween(vectors[klass-1][0], vectors[klass-1][quantity+b]))
		minFalse.append(distanceBetween(vectors[random.randint(0,39)][0], vectors[klass-1][quantity+b]))

	for k in range(quantity,10):
		for s in range(0,quantity):
			if (distanceBetween(vectors[klass-1][k], vectors[klass-1][s]) < minTrue[k-quantity]):
				minTrue[k-quantity] = distanceBetween(vectors[klass-1][k], vectors[klass-1][s])

	for x in range(40):
		for y in range(10):
			if (x == klass - 1):
				break
			for h in range(10-quantity):
				if (distanceBetween(vectors[x][y], vectors[klass-1][h+quantity]) < minFalse[h]):
					#print("Dist of person:" + str(x+1) + "image №: " + str(y) + "to person №" + str(klass) + ", image №" + str(h+quantity)+ " = " + str(distanceBetween(vectors[x][y], vectors[klass-1][h+quantity])))
					minFalse[h] = distanceBetween(vectors[x][y], vectors[klass-1][h+quantity])

	return Persent(minFalse, minTrue)

def CheckDataPersents(vectors, quantity, klass, size):

	minTrue = []
	minFalse = []

	for b in range(10-quantity):
		minTrue.append(distanceBetween(vectors[klass-1][0], vectors[klass-1][quantity+b]))
		minFalse.append(distanceBetween(vectors[random.randint(0,39)][0], vectors[klass-1][quantity+b]))

	for k in range(quantity,10):
		for s in range(0,quantity):
			if (distanceBetween(vectors[klass-1][k], vectors[klass-1][s]) < minTrue[k-quantity]):
				minTrue[k-quantity] = distanceBetween(vectors[klass-1][k], vectors[klass-1][s])

	for x in range(40):
		for y in range(10):
			if (x == klass - 1):
				break
			for h in range(10-quantity):
				if (distanceBetween(vectors[x][y], vectors[klass-1][h+quantity]) < minFalse[h]):
					#print("Dist of person:" + str(x+1) + "image №: " + str(y) + "to person №" + str(klass) + ", image №" + str(h+quantity)+ " = " + str(distanceBetween(vectors[x][y], vectors[klass-1][h+quantity])))
					minFalse[h] = distanceBetween(vectors[x][y], vectors[klass-1][h+quantity])

	return Persent(minFalse, minTrue)

def SearchOptimalParam(toVectorDynamic, ranges):
	vectors = []

	persent = []
	sums = []

	humans = Human()

	for v in ranges:
		for i in range(40):
			vectors.append(toVectorDynamic(humans[i], v))
		for d in range(1,41):
			persent.append(CheckDataPersents(vectors, 9, d, v))
		sums.append(round(sum(persent)/len(persent), 2))
		print(v)
		persent = []
		vectors = []
	fig = plt.figure()
	plt.plot(ranges, sums)
	plt.show()

def SearchQuantityOfPerson(toVectorDynamic, params):
	vectors = []

	persent = []
	sums = []

	humans = Human()

	fig = plt.figure()

	for i in range(40):
		vectors.append(toVectorDynamic(humans[i], params))
	print(str(toVectorDynamic))
	for v in range(0,10):
		for d in range(1,41):
			persent.append(CheckDataPersents(vectors, v, d, params))
		k = random.randint(0,39)
		s = random.randint(0,9)
		f = random.randint(0,9)
		plt.subplot(321),plt.imshow(Human()[k][s],cmap='gray')
		plt.subplot(322),plt.imshow(Human()[k][f],cmap='gray')

		# Гистограмма
		plt.subplot(323),plt.bar(range(0,params),vectors[k][s], color='m')
		plt.subplot(324),plt.bar(range(0,params),vectors[k][f], color='m')

		# Рандомные точки
		# plt.subplot(323),plt.scatter(range(0,params),vectors[k][s], color='m')
		# plt.subplot(324),plt.scatter(range(0,params),vectors[k][f], color='m')

		# Scale
		# plt.subplot(323),plt.imshow(Human()[k][s].resize([math.trunc(92*params),math.trunc(122*params)]),cmap='gray')
		# plt.subplot(324),plt.imshow(Human()[k][f].resize([math.trunc(92*params),math.trunc(122*params)]),cmap='gray')

		# DCT
		# plt.subplot(323),plt.imshow(dct(convert(Human()[k][s].resize([12,12]))),interpolation='nearest', cmap = 'gray')
		# plt.subplot(324),plt.imshow(dct(convert(Human()[k][f].resize([12,12]))),interpolation='nearest', cmap = 'gray')

		# DFT
		# plt.subplot(323),plt.imshow(np.abs(fft(convert(Human()[k][s].resize([12,12])))),interpolation='nearest', cmap = 'gray')
		# plt.subplot(324),plt.imshow(np.abs(fft(convert(Human()[k][f].resize([12,12])))),interpolation='nearest', cmap = 'gray')

		# Gradient
		# l = np.gradient(toList(Human()[k][s].resize([10,10])))
		# g =np.gradient(toList(Human()[k][f].resize([10,10])))


		# plt.subplot(323),plt.plot(range(len(l)),l)
		# plt.subplot(324),plt.plot(range(len(g)),g)

		plt.subplot(313)
		sums.append(round(sum(persent)/len(persent), 2))
		plt.plot(range(0,v+1),sums, color='m')
		plt.pause(0.1)
		plt.clf()
		print(v)
		persent = []

	# print(max(sums), sums.index(max(sums)))
	# plt.plot(range(0,10), sums)
	plt.show()

def SearchQuantityOfAllPerson(toVectorDynamic, params):
	vectors = []

	persent = []
	sums = []

	humans = Human()

	for i in range(40):
		vectors.append(toVectorDynamic(humans[i], params))

	for v in range(0,10):
		for d in range(1,41):
			persent.append(CheckDataPersents(vectors, v, d, params))
		print(v)

	fig = plt.figure()
	plt.plot(range(0,400), persent)
	plt.show()

def Params():
	SearchOptimalParam(toHistogramVectorDynamic, range(2,256))
	SearchOptimalParam(toRandomDynamic, range(1,200))
	SearchOptimalParam(toScaleVectorDynamic, np.arange(0.05, 1, 0.05))
	SearchOptimalParam(toDCTvectorDynamic, range(2,50))
	SearchOptimalParam(toDFTvectorDynamic, range(2,50))
	SearchOptimalParam(toGradVectorDynamicList, range(2, 100))

def Quantity():
	SearchQuantityOfPerson(toHistogramVectorDynamic, 10)
	SearchQuantityOfPerson(toRandomDynamic, 122)
	SearchQuantityOfPerson(toScaleVectorDynamic, 0.4)
	SearchQuantityOfPerson(toDCTvectorDynamic, 12)
	SearchQuantityOfPerson(toDFTvectorDynamic, 16)
	SearchQuantityOfPerson(toGradVectorDynamicList, 10)

def QuantityAll():
	SearchQuantityOfAllPerson(toHistogramVectorDynamic, 10)
	SearchQuantityOfAllPerson(toRandomDynamic, 122)
	SearchQuantityOfAllPerson(toScaleVectorDynamic, 0.4)
	SearchQuantityOfAllPerson(toDCTvectorDynamic, 12)
	SearchQuantityOfAllPerson(toDFTvectorDynamic, 16)
	SearchQuantityOfAllPerson(toGradVectorDynamicList, 10)

# Комментарии для подсчета на всей входящей выборке
def Parallel():
	humans = Human()

	X = 0
	Y = []

	sums = []
	persent = []

	vectorsHist = []
	vectorsRandom = []
	vectorsScale = []
	vectorsDCT = []
	vectorsDFT = []
	vectorsGrad = []

	fig = plt.figure()

	for i in range(40):
		vectorsHist.append(toHistogramVectorDynamic(humans[i], 10))
		vectorsRandom.append(toRandomDynamic(humans[i], 122))
		vectorsScale.append(toScaleVectorDynamic(humans[i], 0.4))
		vectorsDCT.append(toDCTvectorDynamic(humans[i], 12))
		vectorsDFT.append(toDFTvectorDynamic(humans[i], 16))
		vectorsGrad.append(toGradVectorDynamicList(humans[i], 10))

	print("End of Vectors")

	for v in range(0,10):
		print(v)
		for d in range(1,41):
			X = max(MainData(vectorsHist, v, d), MainData(vectorsRandom, v, d), MainData(vectorsScale, v, d), MainData(vectorsDCT, v, d), MainData(vectorsDFT, v, d), MainData(vectorsGrad, v, d))
			persent.append(X)
			# plt.title("Количество человек в обучающей выборке: " + str(v))
			# plt.subplot(211)
			# plt.plot(range(len(persent)), persent)
			# plt.pause(0.0001)
		sums.append(round(sum(persent)/len(persent), 2))
		plt.subplot(212)
		plt.plot(range(len(sums)), sums, c='m')
		plt.pause(0.0001)
		persent = []

	# fig = plt.figure()
	# print(max(sums), sums.index(max(sums)))
	# plt.plot(range(0,400,40), sums)
	plt.show()

points = getRandomPoints()
# SearchQuantityOfPerson(toHistogramVectorDynamic, 10)
# Parallel()
# SearchOptimalParam(toHistogramVectorDynamic, range(2,256))
# SearchQuantityOfPerson(toHistogramVectorDynamic, 10)

def FinalData(vectors, quantity, klass):
	#quantity = [0,10]

	minTrue = []
	minFalse = []

	imgTrue = []
	imgFalse = []

	for b in range(10-quantity):
		rand=random.randint(0,39)
		minTrue.append(distanceBetween(vectors[klass-1][0], vectors[klass-1][quantity+b]))
		minFalse.append(distanceBetween(vectors[rand][0], vectors[klass-1][quantity+b]))
		imgTrue.append(klass-1 + 0.1)
		imgFalse.append(rand + 0.1)



	for k in range(quantity,10):
		for s in range(0,quantity):
			if (distanceBetween(vectors[klass-1][k], vectors[klass-1][s]) < minTrue[k-quantity]):
				minTrue[k-quantity] = distanceBetween(vectors[klass-1][k], vectors[klass-1][s])
				imgTrue[k-quantity] = klass-1 +0.1*s

	for x in range(40):
		for y in range(10):
			if (x == klass - 1):
				break
			for h in range(10-quantity):
				if (distanceBetween(vectors[x][y], vectors[klass-1][h+quantity]) < minFalse[h]):
					#print("Dist of person:" + str(x+1) + "image №: " + str(y) + "to person №" + str(klass) + ", image №" + str(h+quantity)+ " = " + str(distanceBetween(vectors[x][y], vectors[klass-1][h+quantity])))
					minFalse[h] = distanceBetween(vectors[x][y], vectors[klass-1][h+quantity])
					imgFalse[h] = x +0.1*y

	return Klasses(minFalse,minTrue,imgTrue, imgFalse)

def Klasses(arrFalse, arrTrue, imgTrue, imgFalse):
	result = []

	q = len(arrFalse)

	for index in range(q):
		if (arrFalse[index] < arrTrue[index]):
			result.append(parseHum(imgFalse[index]))
		else:
			result.append(parseHum(imgTrue[index]))

	return result

def parseHum(data):
	s = str(data)
	m = s.index('.')
	human = int(s[0:m])
	img = int(s[m+1:m+2])
	return [human, img]

def Score(data, k, klass):
	score = []
	for d in data:
		if (d[k][0]+1 == klass):
			score.append(d[k][0])
	return len(score)/len(data)*100

def param(data, k, klass):
	result = []
	for d in data:
		if(d[k][0] == klass-1):
			result.append(100)
		else:
			result.append(0)
	return result

def Main():

	humans = Human()

	vectorsHist = []
	vectorsRandom = []
	vectorsScale = []
	vectorsDCT = []
	vectorsDFT = []
	vectorsGrad = []

	c = points[0:110]
	x = []
	y = []
	for i in range(110):
		print(c[i])
		y.append(c[i][0])
		x.append(c[i][1])

	h = math.trunc(92*0.4)
	w = math.trunc(112*0.4)

	methods = [vectorsHist, vectorsRandom, vectorsScale, vectorsDCT, vectorsDFT, vectorsGrad]

	for i in range(40):
		vectorsHist.append(toHistogramVectorDynamic(humans[i], 10))
		vectorsRandom.append(toRandomDynamic(humans[i], 122))
		vectorsScale.append(toScaleVectorDynamic(humans[i], 0.4))
		vectorsDCT.append(toDCTvectorDynamic(humans[i], 12))
		vectorsDFT.append(toDFTvectorDynamic(humans[i], 16))
		vectorsGrad.append(toGradVectorDynamicList(humans[i], 10))
	print("End of Vectors")

	X = []
	Y = []
	xt = []
	yt = []
	txt = []

	H = []
	R = []
	S = []
	C = []
	F = []
	G = []

	Sc = []

	P = []

	fig = plt.figure()
	mpl.style.use('seaborn')
	data = []

	for klass in range(1,41):
		for m in range(len(methods)):
			data.append(FinalData(methods[m], 6, klass))
		for k in range(0,4):
			plt.subplot(271),plt.imshow(open('./s' + str(klass)+'/'+str(k+7)+'.pgm'), cmap = 'gray')
			plt.title("Входное изображение"),plt.axis('off')
			for d in range(len(data)):
				if (d == 0):
					plt.subplot(372),plt.imshow(open('./s' + str(data[d][k][0]+1)+'/'+str(data[d][k][1]+1)+'.pgm'), cmap = 'gray')
					plt.title("Histogram"),plt.axis('off')
					plt.subplot(375),plt.bar(range(0,10), vectorsHist[klass-1][k+6])
					plt.title("Histogram"),plt.axis('off')
				if (d == 1):
					plt.subplot(373),plt.imshow(open('./s' + str(data[d][k][0]+1)+'/'+str(data[d][k][1]+1)+'.pgm'), cmap = 'gray')
					plt.title("Random"),plt.axis('off')
					plt.subplot(376),plt.imshow(open('./s' + str(klass)+'/'+str(k+7)+'.pgm'), cmap = 'gray'),plt.scatter(x,y,c='r')
					plt.title("Random"),plt.axis('off')
				if (d == 2):
					plt.subplot(374),plt.imshow(open('./s' + str(data[d][k][0]+1)+'/'+str(data[d][k][1]+1)+'.pgm'), cmap = 'gray')
					plt.title("Scale"),plt.axis('off')
					plt.subplot(377),plt.imshow(open('./s' + str(klass)+'/'+str(k+7)+'.pgm').resize([h,w]), cmap = 'gray')
					plt.title("Scale"),plt.axis('off')
				if (d == 3):
					img = open('./s' + str(data[d][k][0]+1)+'/'+str(data[d][k][1]+1)+'.pgm')
					plt.subplot(379),plt.imshow(img, cmap = 'gray')
					plt.title("DCT"),plt.axis('off')
					plt.subplot(3,7,12),plt.imshow(dct(convert(img.resize([12,12]))),interpolation='nearest', cmap = 'gray')
					plt.title("DCT"),plt.axis('off')
				if (d == 4):
					img = open('./s' + str(data[d][k][0]+1)+'/'+str(data[d][k][1]+1)+'.pgm')
					plt.subplot(3,7,10),plt.imshow(img, cmap = 'gray')
					plt.title("DFT"),plt.axis('off')
					plt.subplot(3,7,13),plt.imshow(np.abs(fft(convert(img.resize([16,16])))),interpolation='nearest', cmap = 'gray')
					plt.title("DFT"),plt.axis('off')
				if (d == 5):
					plt.subplot(3,7,11),plt.imshow(open('./s' + str(data[d][k][0]+1)+'/'+str(data[d][k][1]+1)+'.pgm'), cmap = 'gray')
					plt.title("Gradient"),plt.axis('off')
					plt.subplot(3,7,14),plt.plot(range(len(vectorsGrad[klass-1][k+6])),vectorsGrad[klass-1][k+6])
					plt.title("Gradient"),plt.axis('off')
			print(data)
			scor = Score(data, k, klass)
			P = param(data, k, klass)
			print(P[5])
			if (scor>50):
				scor = 100
			else:
				scor = 0
			if (len(Y)==0):
				Y.append(scor)
				Sc.append(scor)
				H.append(P[0])
				R.append(P[1])
				S.append(P[2])
				C.append(P[3])
				F.append(P[4])
				G.append(P[5])
			else:
				Sc.append(scor)
				Y.append(sum(Sc)/len(Sc))
				H.append((sum(H)+P[0])/(len(H)+1))
				R.append((sum(R)+P[1])/(len(R)+1))
				S.append((sum(S)+P[2])/(len(S)+1))
				C.append((sum(C)+P[3])/(len(C)+1))
				F.append((sum(F)+P[4])/(len(F)+1))
				G.append((sum(G)+P[5])/(len(G)+1))
				print(G[-1], (sum(G)+P[5]), (len(G)+1))
			plt.subplot(313),plt.plot(range(len(Y)), Y, c='navy',label='Final System',linewidth=1)
			# plt.plot(range(len(H)), H, 'C0',label='Histogram',linewidth=0.5),plt.plot(range(len(R)), R, 'C1',label='Random',linewidth=0.5),plt.plot(range(len(S)), S, 'C2',label='Scale',linewidth=0.5),plt.plot(range(len(C)), C, 'C3',label='DCT',linewidth=0.5),plt.plot(range(len(F)), F, 'C4',label='DFT',linewidth=0.5),plt.plot(range(len(G)), G, 'C5',label='Gradient',linewidth=0.5)
			plt.legend(),plt.title("Точность работы классификатора")
			if (scor == 0):
				yt.append(scor)
				xt.append(len(Y)-1)
				txt.append('s' + str(klass)+ '/' + str(k+7))
			for t in range(len(txt)):
				plt.text(xt[t], y[t], txt[t])
			plt.pause(0.01)
			plt.clf()
			# plt.show()
		P = []
		data = []
	print(sum(Y)/len(Y))

Main()


