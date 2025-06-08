from PIL import Image


def remove_gray_background(input_image_path, output_image_path, gray_threshold=55):
    # Görüntüyü aç
    image = Image.open(input_image_path).convert("RGBA")

    # Görüntü piksel verilerini al
    pixels = image.load()

    # Görüntü boyutlarını al
    width, height = image.size

    # Gri arka planı şeffaf yap
    for x in range(width):
        for y in range(height):
            r, g, b, a = pixels[x, y]

            # Gri rengin aralığını kontrol et (r, g, b'nin eşit ve yakın değeri)
            # Burada, r, g, b'nin çok yakın olması durumunda (eşiği artırarak) gri kabul edebiliriz
            if abs(r - g) < gray_threshold and abs(g - b) < gray_threshold and abs(r - b) < gray_threshold:
                # Eğer piksel gri ise, şeffaf yap
                pixels[x, y] = (r, g, b, 0)  # Alpha kanalını 0 yaparak şeffaf yap
            else:
                # Eğer gri değilse, pikselleri olduğu gibi bırak
                continue

    # Şeffaf arka planlı resmi kaydet
    image.save(output_image_path, "PNG")


# Kullanım örneği
remove_gray_background('misir.png', 'misir_no_gray.png')
