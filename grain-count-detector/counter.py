import cv2
import numpy as np

# Görüntüyü RGBA formatında yükle (alfa kanalı da dahil)
image = cv2.imread('misir.jpg', cv2.IMREAD_UNCHANGED)  # şeffaf arka planlı görüntü

# Gri tonlama yapabilmek için renkli kısmı ayırıyoruz (RGB)
image_rgb = image[:, :, :3]

# Maskeyi gri tonlamaya uygulayalım, böylece şeffaf olmayan alanları belirleyelim
gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

# Görüntüdeki gürültüyü azaltmak için Gaussian blur uygula
blurred = cv2.GaussianBlur(gray, (15, 15), 0)

# Görüntüyü ikili (binary) formata dönüştür
_, binary = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY_INV)

# Morphological işlemler (Erozyon ve Genişletme) ile bitişik taneleri ayırma
kernel = np.ones((5, 5), np.uint8)  # Kernel boyutunu isteğe bağlı olarak değiştirebilirsiniz

# Erozyon: Nesneleri küçültür, bitişik taneleri ayırmak için kullanılır
eroded = cv2.erode(binary, kernel, iterations=1)

# Genişletme: Nesneleri genişletir, doğru konturları bulmamızı sağlar
dilated = cv2.dilate(eroded, kernel, iterations=2)

# Konturları tespit et
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Mısır tanelerinin sayısını yazdır
corn_count = len(contours)
print(f'Mısır tanelerinin sayısı: {corn_count}')

# Sonuçları görselleştir
output = image.copy()

# Her bir konturu çiz ve üzerine numara yaz
for i, contour in enumerate(contours, start=1):
    if cv2.contourArea(contour) > 100:  # Küçük alanları dikkate alma
        # Konturun etrafına yeşil renkli çizgi çiz
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)

        # Konturun merkezini bul
        M = cv2.moments(contour)
        if M["m00"] != 0:  # Bölme hatasından kaçınmak için
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Numara yaz
            cv2.putText(output, str(i), (cX - 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Toplam mısır sayısını yazdır
cv2.putText(output, f'Toplam Misir: {corn_count}', (24, 112), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 10)

# Çıktıyı dosyaya kaydet
cv2.imwrite('corn_output_with_numbers.png', output)
