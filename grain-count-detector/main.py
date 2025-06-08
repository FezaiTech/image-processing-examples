from PIL import Image
import cv2
import numpy as np


def remove_gray_background(input_image_path, temp_output_path, gray_threshold=55):
    # Görüntüyü aç ve RGBA formatına çevir
    image = Image.open(input_image_path).convert("RGBA")
    pixels = image.load()
    width, height = image.size

    # Gri arka planı şeffaf yap
    for x in range(width):
        for y in range(height):
            r, g, b, a = pixels[x, y]
            if abs(r - g) < gray_threshold and abs(g - b) < gray_threshold and abs(r - b) < gray_threshold:
                pixels[x, y] = (r, g, b, 0)  # Şeffaf yap
    image.save(temp_output_path, "PNG")


def count_corn_kernels(image_path, output_path):
    # Görüntüyü RGBA formatında oku
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image.shape[2] != 4:
        print("Görüntüde alfa kanalı bulunmuyor!")
        return

    alpha_channel = image[:, :, 3]
    mask = alpha_channel > 0
    image_rgb = image[:, :, :3]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    gray[~mask] = 255  # Şeffaf bölgeleri beyaz yap

    # Gürültü azaltma ve eşikleme
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, binary = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY_INV)

    # Morfolojik işlemler
    kernel = np.ones((5, 5), np.uint8)  # Kernel boyutunu artırarak ayrılmayı kolaylaştırıyoruz
    eroded = cv2.erode(binary, kernel, iterations=1)  # Erozyon: taneleri birbirinden ayır
    dilated = cv2.dilate(eroded, kernel, iterations=2)  # Genişletme: tanelerin sınırlarını belirginleştir

    # Kontur bulma
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Çıkış görseli
    output = image.copy()
    count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX  # Yazı tipi
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 100:  # Küçük alanları yok sayıyoruz
            count += 1
            # Konturun etrafını çiz
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)

            # Konturun merkezini hesapla
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Sayıyı yaz
                cv2.putText(output, str(count), (cX - 10, cY - 10), font, 0.5, (0, 255, 0), 2)

    # Sayıyı göster
    print(f"Mısır tanelerinin sayısı: {count}")

    # Çıktıyı dosyaya kaydet
    cv2.imwrite(output_path, output)


# === Ana akış ===
input_image = 'misir.png'
temp_image = 'misir_no_gray.png'
output_image = 'corn_output.png'

remove_gray_background(input_image, temp_image)
count_corn_kernels(temp_image, output_image)
