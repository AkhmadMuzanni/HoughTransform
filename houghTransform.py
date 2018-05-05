import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('Monas.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
np.set_printoptions(threshold='nan')
baris,kolom = edges.shape
print edges.shape
cv2.namedWindow("Monas.jpg",cv2.WINDOW_NORMAL)
cv2.imshow('Monas.jpg',edges)

def hough_line(img, akurasi_pixel=1):
    arraytheta = np.deg2rad(np.arange(-90.0, 90.0, akurasi_pixel))
    panjangdiagonal = int(np.round(np.hypot(baris, kolom)))
    arrayrho = np.linspace (-panjangdiagonal, panjangdiagonal, panjangdiagonal * 2)
    panjang_theta = len(arraytheta)
    cos_t = np.cos (arraytheta)
    sin_t = np.sin (arraytheta)


    acumulator = np.zeros ((2 * panjangdiagonal, panjang_theta), dtype=np.uint8)
    # (row, col) indexes to edges
    yindeks, xindeks = np.nonzero (img)


    for i in range (len (xindeks)):
        x = xindeks[i]
        y = yindeks[i]

        for t_idx in range (panjang_theta):
            rho = panjangdiagonal + int(round (x * cos_t[t_idx] + y * sin_t[t_idx]))
            acumulator[rho, t_idx] += 1

    return acumulator, arraytheta, arrayrho

def show_hough_line(accumulator, thetas, rhos):
    import matplotlib.pyplot as plt

    plt.imshow(accumulator, cmap='jet',aspect='auto',
               extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]],)
    # plt.axis('off')
    plt.show()

a, t, r = hough_line(edges)
hasil = []
print a
print t
print r


b,k = a.shape


print a.shape
for i in range (b):
    for j in range (k):
        if a[i][j] > 15:
            indexi =np.abs(np.round(r[i]))
            indexj =(t[j])
            hasil.append([(indexi, indexj)])



print hasil
hasilarray = np.asarray(hasil)
print hasilarray

lines = cv2.HoughLines(edges,1,np.pi/180,200)
print lines

for rho,theta in hasilarray[:,0,:]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
cv2.imwrite('best.jpg',img)
print 'finish'

show_hough_line(a,t,r)
# for i in range(a[0]):
#     for

