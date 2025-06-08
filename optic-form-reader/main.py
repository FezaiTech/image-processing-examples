import cv2
import numpy as np


def find_colored_area(image, target_color):
    # Belirtilen renkli alanı bulmak için maske oluştur
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if target_color == (17, 255, 0):  # Yeşil (#11FF00)
        lower = np.array([50, 100, 100])
        upper = np.array([70, 255, 255])
    elif target_color == (255, 221, 0):  # Sarı (#FFDD00)
        lower = np.array([20, 100, 100])
        upper = np.array([30, 255, 255])
    elif target_color == (255, 0, 251):  # Pembe (#FF00FB)
        lower = np.array([140, 100, 100])
        upper = np.array([160, 255, 255])
    elif target_color == (255, 0, 4):  # Kırmızı (#FF0004)
        lower = np.array([0, 100, 100])
        upper = np.array([10, 255, 255])
    elif target_color == (0, 242, 255):  # Mavi (#00F2FF)
        lower = np.array([80, 100, 100])  # Daha geniş mavi aralığı
        upper = np.array([100, 255, 255])
    else:
        raise ValueError("Geçersiz renk!")

    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"{target_color} renkli alan bulunamadı!")

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, w, h


def detect_circles(image, x, y, w, h, expected_count, orientation="vertical"):
    # Belirtilen alanda yuvarlakları tespit et
    roi = image[y:y + h, x:x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Hough Circle parametrelerini optimize et
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=10,  # Daha küçük mesafe, yoğun grid için
        param1=50,
        param2=20,  # Daha hassas tespit
        minRadius=3,
        maxRadius=25
    )

    if circles is None or len(circles[0]) < expected_count:
        raise ValueError(
            f"Yeterli yuvarlak tespit edilemedi! ({len(circles[0]) if circles is not None else 0}/{expected_count})")

    circles = np.uint16(np.around(circles))
    centers = [(circle[0] + x, circle[1] + y) for circle in circles[0, :]]

    # Yuvarlakları sırala
    if expected_count > 4:  # Öğrenci numarası veya cevaplar için grid
        if expected_count == 80:  # Öğrenci numarası (8x10)
            # Önce x'e göre sütunları sırala
            centers = sorted(centers, key=lambda c: c[0])
            grid = []
            for i in range(0, len(centers), 10):
                column = centers[i:i + 10]
                if len(column) != 10:
                    print(f"Uyarı: Öğrenci numarası sütununda {len(column)} yuvarlak bulundu, beklenen: 10")
                column = sorted(column, key=lambda c: c[1])  # Her sütunu y'ye göre sırala
                grid.append(column)
            if len(grid) != 8:
                raise ValueError(f"Öğrenci numarası gridi hatalı! {len(grid)} sütun bulundu, beklenen: 8")
        else:  # Cevaplar (20x5)
            # Önce y'ye göre satırları sırala
            centers = sorted(centers, key=lambda c: c[1])
            grid = []
            for i in range(0, len(centers), 5):
                row = centers[i:i + 5]
                if len(row) != 5:
                    print(f"Uyarı: Cevap satırında {len(row)} yuvarlak bulundu, beklenen: 5")
                row = sorted(row, key=lambda c: c[0])  # Her satırı x'e göre sırala
                grid.append(row)
            if len(grid) != 20:
                raise ValueError(f"Cevap gridi hatalı! {len(grid)} satır bulundu, beklenen: 20")
        return grid
    else:
        # Tek boyutlu sıralama
        if orientation == "vertical":
            centers = sorted(centers, key=lambda c: c[1])
        else:
            centers = sorted(centers, key=lambda c: c[0])
        return centers[:expected_count]


def read_marked_circles(image, centers, threshold=80, is_grid=False):
    if is_grid:
        result = ""
        for row_idx, row in enumerate(centers):
            marked_count = 0
            marked_answer = None
            for idx, (cx, cy) in enumerate(row):
                roi = image[cy - 5:cy + 5, cx - 5:cx + 5]
                if roi.size == 0:
                    continue
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                avg_intensity = np.mean(gray_roi)
                if avg_intensity < threshold:
                    marked_count += 1
                    marked_answer = str(idx) if len(row) == 10 else chr(65 + idx)
            if marked_count > 1:
                result += "M"  # Multiple marks indicator
                print(f"Soru {row_idx + 1}: Birden fazla işaretleme tespit edildi")
            elif marked_count == 0:
                result += "X"
                print(f"Soru {row_idx + 1}: Boş")
            else:
                result += marked_answer
                print(f"Soru {row_idx + 1}: {marked_answer}")
        return result
    else:
        marked_count = 0
        marked_idx = -1
        for idx, (cx, cy) in enumerate(centers):
            roi = image[cy - 5:cy + 5, cx - 5:cx + 5]
            if roi.size == 0:
                continue
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            avg_intensity = np.mean(gray_roi)
            if avg_intensity < threshold:
                marked_count += 1
                marked_idx = idx
        if marked_count > 1:
            return -2  # Multiple marks indicator for non-grid
        return marked_idx


def check_answers(answers, group, answer_key_map):
    if group not in answer_key_map or not answer_key_map[group]:
        return 0, 0, 20
    answer_key = answer_key_map[group]
    correct = 0
    wrong = 0
    blank = 0
    for student_answer, correct_answer in zip(answers, answer_key):
        if student_answer == "X":
            blank += 1
        elif student_answer == "M":
            wrong += 1  # Multiple marks considered wrong
        elif student_answer == correct_answer:
            correct += 1
        else:
            wrong += 1
    return correct, wrong, blank


def main(template_path, optic_path, answer_key_map):
    # Görüntüleri yükle
    template_img = cv2.imread(template_path)
    optic_img = cv2.imread(optic_path)

    if template_img is None or optic_img is None:
        raise ValueError("Görüntü dosyaları yüklenemedi!")

    # Yeşil alan: Öğrenci numarası (8x10)
    print("Yeşil alan (öğrenci numarası) işleniyor...")
    x, y, w, h = find_colored_area(template_img, (17, 255, 0))
    student_grid = detect_circles(template_img, x, y, w, h, 80, "vertical")
    student_number = read_marked_circles(optic_img, student_grid, is_grid=True)

    # Sarı alan: Sınav türü (4 yuvarlak, alt alta)
    print("Sarı alan (sınav türü) işleniyor...")
    x, y, w, h = find_colored_area(template_img, (255, 221, 0))
    exam_type_circles = detect_circles(template_img, x, y, w, h, 4, "vertical")
    exam_type_idx = read_marked_circles(optic_img, exam_type_circles)
    exam_types = ["Ara Sınav", "Yarıyıl Sonu", "Bütünleme", "Diğer"]
    exam_type = exam_types[exam_type_idx] if exam_type_idx >= 0 else "Bilinmiyor"

    # Pembe alan: Grup numarası (4 yuvarlak, yan yana)
    print("Pembe alan (grup numarası) işleniyor...")
    x, y, w, h = find_colored_area(template_img, (255, 0, 251))
    group_circles = detect_circles(template_img, x, y, w, h, 4, "horizontal")
    group_idx = read_marked_circles(optic_img, group_circles)
    groups = ["A", "B", "C", "D"]
    group = groups[group_idx] if group_idx >= 0 else "Bilinmiyor"

    # Kırmızı alan: Dönem (3 yuvarlak, yan yana)
    print("Kırmızı alan (dönem) işleniyor...")
    x, y, w, h = find_colored_area(template_img, (255, 0, 4))
    semester_circles = detect_circles(template_img, x, y, w, h, 3, "horizontal")
    semester_idx = read_marked_circles(optic_img, semester_circles)
    semesters = ["Güz", "Bahar", "Yaz Okulu"]
    semester = semesters[semester_idx] if semester_idx >= 0 else "Bilinmiyor"

    # Mavi alan: Cevaplar (20x5)
    print("Mavi alan (cevaplar) işleniyor...")
    x, y, w, h = find_colored_area(template_img, (0, 242, 255))
    answer_grid = detect_circles(template_img, x, y, w, h, 100, "vertical")
    answers = read_marked_circles(optic_img, answer_grid, is_grid=True)

    # Cevapları kontrol et
    correct, wrong, blank = check_answers(answers, group, answer_key_map)

    # Sonuçları yazdır
    print("\nSonuçlar:")
    print("Öğrenci Numarası:", student_number)
    print("Sınav Türü:", exam_type)
    print("Grup:", group)
    print("Dönem:", semester)
    print("Cevaplar:", answers)
    print("Doğru Cevap Sayısı:", correct)
    print("Yanlış Cevap Sayısı:", wrong)
    print("Boş Cevap Sayısı:", blank)


if __name__ == "__main__":
    # Cevap anahtarı map'i
    answer_key_map = {
        "A": ["A", "C", "D", "E", "E", "D", "A", "E", "A", "C", "A", "B", "C", "C", "D", "E", "E", "B", "A", "D"],
        "B": [],
        "C": [],
        "D": []
    }

    template_path = "template.png"
    optic_path = "./optics/21486559.png"
    main(template_path, optic_path, answer_key_map)