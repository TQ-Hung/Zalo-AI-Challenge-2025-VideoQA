import json
import csv

# Đọc file JSON (đổi tên đúng file của bạn)
with open("output/submission.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Mapping số -> chữ
idx2label = {0: "A", 1: "B", 2: "C", 3: "D"}

# Ghi ra CSV
with open("output/submission.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "answer"])
    for item in data:
        ans = item["answer"]
        if isinstance(ans, int):
            ans = idx2label.get(ans, "A")  # mặc định A nếu lỗi
        writer.writerow([item["id"], ans])

print("✅ Đã tạo file submission.csv thành công!")
