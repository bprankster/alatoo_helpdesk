Place official university PDFs here.

Recommended files to add:
- admissions_guide_2025.pdf       — Official admissions booklet
- faculty_engineering_2025.pdf    — Engineering & IT faculty catalog
- faculty_economics_2025.pdf      — Economics & Management faculty catalog
- faculty_medicine_2025.pdf       — Medical faculty catalog
- faculty_humanities_2025.pdf     — Humanities faculty catalog
- faculty_social_2025.pdf         — Social Sciences faculty catalog
- ort_discounts_2025.pdf          — Official ORT discount table
- tuition_fees_2025.pdf           — Tuition fee schedule

PDF filenames should contain faculty keywords for auto-detection:
  engineering / informatics → Факультет инженерии и информатики
  economics / экономик      → Факультет экономики и управления
  medicine / медицин        → Медицинский факультет
  social / социал           → Факультет социальных наук
  human / гуманит / lingv   → Факультет гуманитарных наук

After adding PDFs, re-run ingestion:
  python data_ingestion/embedder.py
