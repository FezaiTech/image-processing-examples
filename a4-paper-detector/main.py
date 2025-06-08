import cv2
import numpy as np

A4_WIDTH = 397
A4_HEIGHT = 562

# Çıktıyı kaydederle a4 boyutunda gösteriyoruz
def img_save_show(img, name, winname):
    cv2.imwrite(name, img)
    img_a4 = cv2.resize(img, (A4_WIDTH, A4_HEIGHT))
    cv2.imshow(winname, img_a4)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)

# Görselimiz OpenCv Kütüphnesi ile okuyoruz
img = cv2.imread("evrak4.jpg")
if img is None:
    raise Exception("Görsel dosyası okunamadı! Dosya yolunu kontrol et.")

# 2. Griye çeviriyoruz
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_save_show(gray, "step_1_gray.jpg", "Adım 1: Gri Görüntü")

# 3. Blur uyguluyoruz
blur = cv2.GaussianBlur(gray, (5, 5), 0)
img_save_show(blur, "step_2_blur.jpg", "Adım 2: Gaussian Blur")

# 4. (Canny) Kenar buluyoruz 
edges = cv2.Canny(blur, 75, 200)
img_save_show(edges, "step_3_edges.jpg", "Adım 3: Kenar Tespiti (Canny)")

# 5. Kontur buluyoruz
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

doc_cnt = None
for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4:
        doc_cnt = approx
        break

if doc_cnt is None:
    raise Exception("Belge konturu bulunamadı!")

# Belge kenarlarını orijinal görüntü üzerinde yeşil border çiziyoruz
img_with_contour = img.copy()
cv2.drawContours(img_with_contour, [doc_cnt], -1, (0, 255, 0), 5)  # yeşil çizgi
img_save_show(img_with_contour, "step_3.5_green_contour.jpg", "Adım 3.5: Belge Kenarları (Yeşil Çizgi)")

# 6. Mask uyguluyoruz
mask = np.zeros_like(gray)
cv2.drawContours(mask, [doc_cnt], -1, 255, -1)
masked = cv2.bitwise_and(img, img, mask=mask)
img_save_show(masked, "step_4_masked.jpg", "Adım 4: Maskeli Görüntü")

# 7. Belge bölgesini kırpıyoruz
x, y, w, h = cv2.boundingRect(doc_cnt)
cropped = masked[y:y+h, x:x+w]
img_save_show(cropped, "step_5_cropped.jpg", "Adım 5: Kırpılmış Görüntü")

# 8. Griye çevirme ve kontrast artırma işlemi uyguluyoruz
gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
contrast_img = clahe.apply(gray_cropped)
img_save_show(contrast_img, "step_6_contrast.jpg", "Adım 6: Kontrast Artırılmış Görüntü")

# 9. Adaptive threshold ile arka planı beyazlatıyoruz
thresh = cv2.adaptiveThreshold(
    contrast_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)
img_save_show(thresh, "step_7_thresh.jpg", "Adım 7: Adaptive Threshold (Beyaz Zemin)")

# 10. Perspektif düzeltiyoruz
def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

pts = doc_cnt.reshape(4, 2)
rect = order_points(pts)
(tl, tr, br, bl) = rect

widthA = np.linalg.norm(br - bl)
widthB = np.linalg.norm(tr - tl)
maxWidth = max(int(widthA), int(widthB))

heightA = np.linalg.norm(tr - br)
heightB = np.linalg.norm(tl - bl)
maxHeight = max(int(heightA), int(heightB))

dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")

M = cv2.getPerspectiveTransform(rect, dst)
warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
img_save_show(warped, "output_warped.jpg", "Adım 8: Perspektif Düzeltme (Sonuç)")

print("Tüm adımlar başarıyla kaydedildi ve A4 boyutunda ekranda gösterildi.")