import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from collections import defaultdict


def find_colored_area(image, target_color):
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
        lower = np.array([80, 100, 100])
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
    roi = image[y:y + h, x:x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=10,
        param1=50,
        param2=20,
        minRadius=3,
        maxRadius=25
    )

    if circles is None or len(circles[0]) < expected_count:
        raise ValueError(
            f"Yeterli yuvarlak tespit edilemedi! ({len(circles[0]) if circles is not None else 0}/{expected_count})")

    circles = np.uint16(np.around(circles))
    centers = [(circle[0] + x, circle[1] + y) for circle in circles[0, :]]

    if expected_count > 4:
        if expected_count == 80:
            centers = sorted(centers, key=lambda c: c[0])
            grid = []
            for i in range(0, len(centers), 10):
                column = centers[i:i + 10]
                if len(column) != 10:
                    print(f"Uyarı: Öğrenci numarası sütununda {len(column)} yuvarlak bulundu, beklenen: 10")
                column = sorted(column, key=lambda c: c[1])
                grid.append(column)
            if len(grid) != 8:
                raise ValueError(f"Öğrenci numarası gridi hatalı! {len(grid)} sütun bulundu, beklenen: 8")
        else:
            centers = sorted(centers, key=lambda c: c[1])
            grid = []
            for i in range(0, len(centers), 5):
                row = centers[i:i + 5]
                if len(row) != 5:
                    print(f"Uyarı: Cevap satırında {len(row)} yuvarlak bulundu, beklenen: 5")
                row = sorted(row, key=lambda c: c[0])
                grid.append(row)
            if len(grid) != 20:
                raise ValueError(f"Cevap gridi hatalı! {len(grid)} satır bulundu, beklenen: 20")
        return grid
    else:
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


def process_optic_form(template_path, optic_path, answer_key_map):
    template_img = cv2.imread(template_path)
    optic_img = cv2.imread(optic_path)

    if template_img is None or optic_img is None:
        raise ValueError(f"Görüntü dosyaları yüklenemedi: {optic_path}")

    # Process student number (Green area)
    x, y, w, h = find_colored_area(template_img, (17, 255, 0))
    student_grid = detect_circles(template_img, x, y, w, h, 80, "vertical")
    student_number = read_marked_circles(optic_img, student_grid, is_grid=True)

    # Process exam type (Yellow area)
    x, y, w, h = find_colored_area(template_img, (255, 221, 0))
    exam_type_circles = detect_circles(template_img, x, y, w, h, 4, "vertical")
    exam_type_idx = read_marked_circles(optic_img, exam_type_circles)
    exam_types = ["Ara Sınav", "Yarıyıl Sonu", "Bütünleme", "Diğer"]
    exam_type = exam_types[exam_type_idx] if exam_type_idx >= 0 else "Bilinmiyor"

    # Process group (Pink area)
    x, y, w, h = find_colored_area(template_img, (255, 0, 251))
    group_circles = detect_circles(template_img, x, y, w, h, 4, "horizontal")
    group_idx = read_marked_circles(optic_img, group_circles)
    groups = ["A", "B", "C", "D"]
    group = groups[group_idx] if group_idx >= 0 else "Bilinmiyor"

    # Process semester (Red area)
    x, y, w, h = find_colored_area(template_img, (255, 0, 4))
    semester_circles = detect_circles(template_img, x, y, w, h, 3, "horizontal")
    semester_idx = read_marked_circles(optic_img, semester_circles)
    semesters = ["Güz", "Bahar", "Yaz Okulu"]
    semester = semesters[semester_idx] if semester_idx >= 0 else "Bilinmiyor"

    # Process answers (Blue area)
    x, y, w, h = find_colored_area(template_img, (0, 242, 255))
    answer_grid = detect_circles(template_img, x, y, w, h, 100, "vertical")
    answers = read_marked_circles(optic_img, answer_grid, is_grid=True)

    # Check answers
    correct, wrong, blank = check_answers(answers, group, answer_key_map)

    return {
        "student_number": student_number,
        "exam_type": exam_type,
        "group": group,
        "semester": semester,
        "answers": answers,
        "correct": correct,
        "wrong": wrong,
        "blank": blank,
        "file_name": os.path.basename(optic_path)
    }


class AnswerKeyWindow(tk.Toplevel):
    def __init__(self, parent, callback):
        super().__init__(parent)
        self.title("Cevap Anahtarı Girişi")
        self.geometry("400x600")
        self.callback = callback
        self.answer_key = []
        self.group = tk.StringVar(value="A")

        tk.Label(self, text="Grup Seçimi:").pack(pady=5)
        tk.OptionMenu(self, self.group, "A", "B", "C", "D").pack()

        self.entries = []
        for i in range(20):
            frame = tk.Frame(self)
            frame.pack(pady=2)
            tk.Label(frame, text=f"Soru {i+1}:", width=10).pack(side=tk.LEFT)
            entry = ttk.Combobox(frame, values=["A", "B", "C", "D", "E"], width=5)
            entry.pack(side=tk.LEFT)
            self.entries.append(entry)

        tk.Button(self, text="Kaydet", command=self.save).pack(pady=10)

    def save(self):
        answers = [entry.get() for entry in self.entries]
        if all(answer in ["A", "B", "C", "D", "E"] for answer in answers):
            self.callback(self.group.get(), answers)
            self.destroy()
        else:
            messagebox.showerror("Hata", "Tüm sorular için geçerli bir cevap (A-E) seçiniz!")


class OpticalFormScanner(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Optik Form Tarayıcı")
        self.geometry("800x600")
        self.template_path = None
        self.optic_paths = []
        self.answer_key_map = defaultdict(list)

        # Template selection
        tk.Button(self, text="Template Seç", command=self.select_template).pack(pady=5)
        self.template_label = tk.Label(self, text="Template seçilmedi")
        self.template_label.pack()

        # Optic form selection
        tk.Button(self, text="Optik Form(lar) Seç", command=self.select_optic_forms, state=tk.DISABLED).pack(pady=5)
        self.optic_listbox = tk.Listbox(self, height=5, width=60)
        self.optic_listbox.pack(pady=5)

        # Answer key
        tk.Button(self, text="Cevap Anahtarı Ekle", command=self.add_answer_key).pack(pady=5)

        # Cevap anahtarı metin alanı
        self.answer_key_display = tk.Text(self, height=5, width=60)  # Use a Text widget to display multiple lines
        self.answer_key_display.pack(pady=5)
        self.answer_key_display.config(state=tk.DISABLED)

        # Scan button
        tk.Button(self, text="Tarama Yap", command=self.scan_forms, state=tk.DISABLED).pack(pady=5)

        # Results display
        self.result_text = tk.Text(self, height=20, width=60)
        self.result_text.pack(pady=5)
        self.result_text.config(state=tk.DISABLED)

    def select_template(self):
        path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
        if path:
            self.template_path = path
            self.template_label.config(text=f"Template: {os.path.basename(path)}")
            self.optic_listbox.delete(0, tk.END)
            self.optic_paths = []
            self.children["!button2"].config(state=tk.NORMAL)
            self.children["!button4"].config(state=tk.DISABLED)

    def select_optic_forms(self):
        paths = filedialog.askopenfilenames(filetypes=[("PNG files", "*.png")])
        if paths:
            self.optic_listbox.delete(0, tk.END)
            self.optic_paths = list(paths)
            for path in self.optic_paths:
                self.optic_listbox.insert(tk.END, os.path.basename(path))
            if self.answer_key_map:
                self.children["!button4"].config(state=tk.NORMAL)

    def add_answer_key(self):
        AnswerKeyWindow(self, self.save_answer_key)

    def save_answer_key(self, group, answers):
        self.answer_key_map[group] = answers
        # Display the answer key information in the Text widget
        self.answer_key_display.config(state=tk.NORMAL)  # Enable text editing to append
        self.answer_key_display.insert(tk.END, f"Grup {group}: {', '.join(answers)}\n")  # Append new answer key info
        self.answer_key_display.config(state=tk.DISABLED)  # Disable text editing again
        if self.optic_paths:
            self.children["!button4"].config(state=tk.NORMAL)
        messagebox.showinfo("Başarılı", f"{group} grubu için cevap anahtarı kaydedildi")

    def scan_forms(self):
        if not self.template_path or not self.optic_paths:
            messagebox.showerror("Hata", "Template ve en az bir optik form seçilmelidir!")
            return

        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)

        for optic_path in self.optic_paths:
            try:
                result = process_optic_form(self.template_path, optic_path, self.answer_key_map)
                self.result_text.insert(tk.END, f"\nDosya: {result['file_name']}\n")
                self.result_text.insert(tk.END, f"Öğrenci Numarası: {result['student_number']}\n")
                self.result_text.insert(tk.END, f"Sınav Türü: {result['exam_type']}\n")
                self.result_text.insert(tk.END, f"Grup: {result['group']}\n")
                self.result_text.insert(tk.END, f"Dönem: {result['semester']}\n")
                self.result_text.insert(tk.END, f"Cevaplar: {result['answers']}\n")
                self.result_text.insert(tk.END,
                                        f"Doğru: {result['correct']}, Yanlış: {result['wrong']}, Boş: {result['blank']}\n")
                self.result_text.insert(tk.END, "-" * 50 + "\n")
            except Exception as e:
                self.result_text.insert(tk.END, f"\nDosya: {os.path.basename(optic_path)}\n")
                self.result_text.insert(tk.END, f"Hata: {str(e)}\n")
                self.result_text.insert(tk.END, "-" * 50 + "\n")

        self.result_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    app = OpticalFormScanner()
    app.mainloop()