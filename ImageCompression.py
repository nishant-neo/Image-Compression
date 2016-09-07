import scipy as sp
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

img = Image.open('Iasp.png')
imggray = img.convert('LA')
plt.figure(figsize = ( 9, 6))
plt.imshow(imggray)

imgmat = np.array( list(imggray.getdata(band = 0)), float)
imgmat.shape = (imggray.size[1], imggray.size[0])
imgmat = np.matrix(imgmat)
plt.figure(figsize=(9,6))
plt.imshow(imgmat, cmap = 'gray')
plt.show()

U, S, Vt = np.linalg.svd(imgmat) #single value decomposition
for i in xrange(5, 51, 5):
	cmpimg = np.matrix(U[:, :i]) * np.diag(S[:i]) * np.matrix(Vt[:i,:])
	plt.imshow(cmpimg, cmap = 'gray')
	title = " n = %s" %i
	plt.title(title)
	plt.show()
	result = Image.fromarray((cmpimg ).astype(np.uint8))
	
result.save('out.png')
	
