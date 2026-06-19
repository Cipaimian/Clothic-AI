# Clothic AI: Sistem Pemantauan Kepatuhan Berpakaian Kampus yang Dapat Dijelaskan Berbasis Pengenalan Atribut Visual dan Mesin Penalaran Kebijakan

**Penulis:** Suteja, Putra, Brian Filbert Chandra, Bagaskara, Athar
**Dosen Pembimbing:** Irvan Santoso, S.Kom., M.TI.
**Program Studi Cyber Security, Universitas Bina Nusantara (BINUS)**

---

## ABSTRAK

Pemantauan kepatuhan tata busana (*dress code*) di lingkungan kampus umumnya
dilakukan secara manual sehingga tidak konsisten, subjektif, dan tidak dapat
diskalakan. Upaya otomasi dengan model *deep learning* tunggal yang dilatih untuk
mengklasifikasikan label biner "sopan/tidak sopan" - sebagaimana pada prototipe
awal kami, **baseline** - terbukti rapuh: model mempelajari penilaian nilai yang
subjektif, tidak dapat dijelaskan, sulit diaudit, dan berisiko bias terhadap
kelompok tertentu. Penelitian ini mengusulkan **Clothic AI** (*Clothing Vision*),
sebuah perancangan ulang menyeluruh menjadi **mesin Pengenalan Atribut Visual +
Penalaran Kebijakan** (*Visual Attribute Recognition + Policy Reasoning Engine*).
Prinsip utamanya: **jaringan saraf tidak pernah mempelajari konsep "kesopanan"** -
jaringan hanya mendeteksi atribut visual yang teramati dan dapat dipertanggung-
jawabkan (jenis garmen, panjang lengan, batas bawah pakaian terhadap lutut, rasio
keterbukaan kulit per region tubuh), sementara sebuah **mesin aturan berbasis JSON
yang transparan** memetakan atribut tersebut menjadi keputusan kepatuhan beserta
penjelasan tertulis dan kutipan peraturan. Sistem mengganti label biner dengan
**empat skor terkalibrasi** (keterbukaan, formalitas, kepatuhan, ketidakpastian)
dan, di bawah kondisi tertutup/buram, memilih untuk **abstain**
(`insufficient_evidence`) alih-alih menuduh secara keliru. Pengukuran keterbukaan
bersifat **geometris berbasis piksel** (parsing tubuh Meta Sapiens ∩ region dari
*pose*), bukan berbasis warna kulit, sehingga lebih adil. Sistem telah
diimplementasikan secara penuh (perangkat reasoning, REST API, antarmuka web)
dengan 60 pengujian otomatis lulus, serta diverifikasi *end-to-end* pada citra
nyata. Kontribusi utama adalah arsitektur yang **dapat dijelaskan, dapat
dikonfigurasi tanpa pelatihan ulang, dapat digugat, dan adil**, yang mengubah
sistem pengawasan kabur (*black box*) menjadi alat bantu kebijakan yang akuntabel.

**Kata kunci:** *explainable AI*, kepatuhan tata busana, pengenalan atribut
visual, mesin aturan, *human parsing*, keadilan algoritmik, abstensi berbasis
ketidakpastian.

## ABSTRACT

Manual enforcement of campus dress codes is inconsistent, subjective, and not
scalable. Automating it with a single deep-learning model trained on a binary
"appropriate/inappropriate" label - as in our earlier **baseline** prototype - is
fragile: the model learns a subjective value judgment, is unexplainable, hard to
audit, and prone to bias. We present **Clothic AI** (*Clothing Vision*), a redesign
into a **Visual Attribute Recognition + Policy Reasoning Engine**. The core
principle is that the neural networks **never learn "modesty"**; they only detect
defensible, observable visual attributes (garment type, sleeve length, hemline
relative to the knee, per-region skin-exposure ratio), while a **transparent
JSON-driven rule engine** maps those attributes to a compliance decision with a
written, citable explanation. The binary label is replaced by **four calibrated
scores** (exposure, formality, compliance, uncertainty), and under occlusion the
system **abstains** (`insufficient_evidence`) rather than risk a false accusation.
Exposure is measured **geometrically at the pixel level** (Meta Sapiens body
parsing ∩ pose-derived regions), not from skin color, improving fairness. The
system is fully implemented (reasoning core, REST API, web UI) with 60 passing
automated tests and verified end-to-end on real images. The main contribution is
an architecture that is **explainable, reconfigurable without retraining,
contestable, and fair**, turning an opaque surveillance tool into an accountable
policy-assistance system.

**Keywords:** explainable AI, dress-code compliance, visual attribute
recognition, rule engine, human parsing, algorithmic fairness, uncertainty-aware
abstention.

---

## 1. PENDAHULUAN

### 1.1 Latar Belakang

Banyak institusi pendidikan menerapkan tata busana (*dress code*) sebagai bagian
dari kode etik akademik. Penegakannya selama ini bersifat manual oleh petugas
atau dosen, yang menimbulkan tiga masalah mendasar: (1) **subjektivitas** -
penilaian "sopan" berbeda antarpenilai; (2) **inkonsistensi** - keputusan
bergantung pada suasana hati, waktu, dan lokasi; serta (3) **keterbatasan skala** -
mustahil mengawasi seluruh titik masuk secara serempak.

Solusi otomasi yang intuitif adalah melatih satu model *computer vision* untuk
mengklasifikasikan citra mahasiswa menjadi "sopan" atau "tidak sopan". Pendekatan
inilah yang ditempuh prototipe awal kami, **baseline**, menggunakan satu model YOLO
tunggal. Namun pendekatan ini memindahkan keputusan nilai yang subjektif ke dalam
bobot jaringan yang tidak transparan, sehingga: keputusan **tidak dapat
dijelaskan** kepada mahasiswa yang terdampak, **tidak dapat digugat**, **tidak
dapat diaudit**, **terkunci** pada satu definisi kesopanan (mengubah aturan berarti
melatih ulang seluruh model), dan **berisiko bias** karena dapat secara tak sengaja
mengaitkan keputusan dengan warna kulit, gender, atau pakaian keagamaan.

### 1.2 Rumusan Masalah

Bagaimana merancang sistem pemantauan kepatuhan berpakaian yang **akurat namun
tetap dapat dijelaskan, dapat dikonfigurasi tanpa pelatihan ulang, dapat digugat
oleh subjek, dan adil terhadap atribut yang dilindungi (gender/etnis)**?

### 1.3 Tujuan dan Kontribusi

Penelitian ini mengusulkan dan mengimplementasikan **Clothic AI**, dengan kontribusi:

1. **Pemisahan persepsi dari penilaian.** Jaringan saraf hanya mendeteksi atribut
   visual objektif; sebuah mesin aturan transparan yang memutuskan kepatuhan.
   Model tidak pernah mempelajari "kesopanan".
2. **Pengukuran keterbukaan berbasis piksel** dengan *human parsing* (Meta
   Sapiens) yang memisahkan kelas anatomi telanjang dan kelas pakaian, sehingga
   keterbukaan = `piksel_kulit / (piksel_kulit + piksel_pakaian)` per region -
   **invarian terhadap warna kulit**.
3. **Metodologi empat skor** (keterbukaan, formalitas, kepatuhan, ketidakpastian)
   yang menggantikan label biner, lengkap dengan **abstensi sadar-ketidakpastian**.
4. **Kebijakan sebagai konfigurasi** - aturan kampus berada dalam berkas JSON yang
   dapat disunting tanpa mengubah kode atau melatih ulang model.
5. **Penjelasan yang dapat digugat** - setiap keputusan menyertakan bukti, aturan
   yang aktif, kutipan peraturan, dan *counterfactual* terverifikasi ("agar patuh:
   …").

---

## 2. TINJAUAN PUSTAKA DAN POSISI PENELITIAN

Pendekatan klasifikasi citra ujung-ke-ujung (mis. CNN/YOLO untuk label tunggal)
unggul untuk objek yang terdefinisi jelas, tetapi tidak sesuai untuk **konsep
normatif** seperti "sopan", yang merupakan keputusan kebijakan, bukan fakta visual.
Clothic AI menyandingkan dua tradisi: (a) **persepsi visual modern** - deteksi objek
(*Ultralytics* YOLO), pelacakan (*ByteTrack*), estimasi *pose*, *human parsing*
(Meta Sapiens), dan model visi-bahasa fesyen (*Marqo-FashionSigLIP*, berbasis
SigLIP-2); dengan (b) **sistem berbasis aturan yang transparan** (*rule engine*)
dan **kalibrasi kepercayaan** (*temperature scaling*, ECE). Penggabungan ini
sejalan dengan prinsip *Explainable AI*: keputusan berisiko terhadap manusia harus
dapat ditelusuri ke bukti yang dapat diperiksa.

Perbedaan utama Clothic AI dari prototipe baseline diringkas pada Tabel 1.

**Tabel 1. Perbandingan baseline (lama) vs Clothic AI (usulan)**

| Aspek | baseline (lama) | Clothic AI (usulan) |
|---|---|---|
| Keluaran | Label biner sopan/tidak | Empat skor + pita keputusan |
| Yang dipelajari model | "Kesopanan" (subjektif) | Atribut visual objektif |
| Penjelasan | Tidak ada | Bukti + aturan + kutipan + *counterfactual* |
| Ubah aturan | Latih ulang model | Sunting JSON, tanpa latih ulang |
| Keterbukaan | Implisit/warna kulit | Piksel: kulit/(kulit+pakaian) per region |
| Ketidakpastian | Dipaksa memutuskan | Abstain (`insufficient_evidence`) |
| Audit & gugatan | Sulit | Tercatat di basis data + jalur banding |

---

## 3. METODE PENELITIAN

### 3.1 Arsitektur Sistem

Clothic AI adalah *pipeline* modular bertahap: **Persepsi → Penalaran → Presentasi**.
Setiap tahap adalah komponen independen dengan kontrak data bertipe (*typed
contract*, pydantic), sehingga model apa pun dapat ditingkatkan tanpa menyentuh
bagian lain.

```
Citra
  → PERSEPSI: deteksi orang → pelacakan → [pose + human parsing]
             → deteksi garmen → klasifikasi atribut → estimasi keterbukaan
  → FUSI TEMPORAL: penghalusan EMA + debounce K-dari-M per track
  → PENALARAN: mesin aturan (JSON) → empat skor → pita keputusan
  → PENJELASAN: narasi templat deterministik + kutipan + counterfactual
  → FrameResult → API / UI Web / basis data audit
```

### 3.2 Persepsi Dua Tahap

Sistem mengikuti pola **"parse tubuh dahulu, lalu periksa apakah garmen menutupi
bagian yang terpindai"**, melalui dua model utama:

- **Model 1 - Pemetaan Tubuh (Meta Sapiens, Goliath 28 kelas).** Sapiens memberi
  label terpisah untuk **anatomi telanjang** (torso, lengan atas/bawah, paha,
  betis, dll.) dan **pakaian** (*Upper/Lower Clothing*, *Apparel*, *Shoe*). Maka
  keterbukaan dapat diukur langsung di tingkat piksel:

  ```
  keterbukaan(region) = piksel_kulit / (piksel_kulit + piksel_pakaian)
  penutupan(region)   = 1 − keterbukaan(region)
  ```

  Jendela region (bahu, lengan atas, perut, paha, lutut, betis) ditempatkan secara
  anatomis menggunakan **anchor dari estimasi *pose*** sehingga pengukuran tepat
  sasaran. Tersedia *fallback* ringan `SegFormer` (`mattmdjaga/segformer_b2_clothes`)
  dengan kontrak keluaran yang sama.

- **Model 2 - Garmen.** *Detector* YOLO terlatih (13 kelas) menentukan **jenis**
  garmen, sedangkan **Marqo-FashionSigLIP** (model SigLIP-2 fine-tuned fesyen)
  menentukan **atribut multi-label** (tanpa lengan, sobek, transparan, *crop*,
  formal) secara *zero-shot* melalui kontras *prompt* teks positif vs negatif.

### 3.3 Mesin Aturan Berbasis JSON

Setiap aturan kampus dinyatakan sebagai predikat yang dapat disunting:

```json
{
  "id": "UPPER_SLEEVELESS",
  "description": "Pakaian tanpa lengan / tank top tidak diperkenankan",
  "severity": 0.7, "weight": 1.0,
  "citation": "Buku Pedoman Mahasiswa §4.2(a)",
  "when": { "attr": "upper.sleeveless", "op": ">=", "value": 0.6 }
}
```

Tata bahasa predikat mendukung `all` / `any` / `not` dan daun
`{attr, op, value}` dengan operator `> >= < <= == != in not_in exists` atas jalur
atribut bertitik (`upper.sleeveless`, `exposure.thigh`, `footwear.type`). Aturan
dapat diaktifkan per **zona** (mis. `lab`, `workshop`). Penelitian ini menyediakan
tiga **profil kebijakan**: `default` (standar kampus), `exam_formal` (ujian/acara
formal), dan `lab_safety` (keselamatan laboratorium, lebih ketat).

### 3.4 Metodologi Empat Skor dan Pita Keputusan

Label biner digantikan empat angka yang dapat ditafsirkan:

| Skor | Makna | Formulasi inti |
|---|---|---|
| `exposure_score` | seberapa jauh keterbukaan region melampaui batas kebijakan | `max_region max(0, e−limit)/(1−limit)` |
| `formality_score` | seberapa formal busana terbaca | basis + atribut formal − sobek + alas kaki |
| `compliance_score` | 1 − magnitudo pelanggaran tersaturasi | `1 − (1 − e^{−k·Σ w·severity})` |
| `uncertainty_score` | 1 − kualitas bukti observasi | `1 − evidence_quality` |
| `overall_violation` | magnitudo pelanggaran utama (`null` bila tidak pasti) | - |

Magnitudo pelanggaran menggunakan pemetaan **menjenuh** (*saturating*),
`1 − exp(−k·Σ weight·severity)`, agar penumpukan banyak pelanggaran ringan tidak
melampaui pelanggaran tunggal yang berat secara tak terkendali. Keputusan akhir
ditentukan oleh **pita keputusan** per profil: jika `uncertainty` melampaui ambang
maka sistem **abstain** (`insufficient_evidence`); jika tidak, dibandingkan
terhadap ambang `compliant` / `minor_violation` / `major_violation`.

### 3.5 Fusi Temporal, Kalibrasi, dan Abstensi

Untuk aliran video, observasi per *track* dihaluskan dengan **EMA** dan keputusan
distabilkan dengan **debounce K-dari-M** agar tidak berkedip. Skor kepercayaan
setiap kepala model dikalibrasi dengan **temperature scaling** (dievaluasi dengan
*Expected Calibration Error*, ECE) sehingga angka "0,87" benar-benar bermakna 87%.
Pita `insufficient_evidence` adalah buah dari kalibrasi: di bawah oklusi, pose
ekstrem, atau pencahayaan buruk, sistem **menolak memutuskan** alih-alih menebak -
properti yang sekaligus lebih adil dan lebih andal.

### 3.6 Penjelasan dan *Counterfactual*

Setiap keputusan menghasilkan: (a) peta bukti (atribut + nilai keterbukaan per
region), (b) daftar aturan yang aktif beserta kutipan peraturan, dan (c)
**counterfactual terverifikasi** - langkah konkret agar patuh (mis. "kenakan atasan
berlengan") yang **diuji ulang terhadap mesin aturan** untuk memastikan langkah itu
benar-benar mengubah verdikt menjadi patuh. Penjelasan bersifat **templat
deterministik** (bukan dihasilkan model bahasa) demi keterulangan dan akuntabilitas.

### 3.7 Dataset dan Pelatihan

*Detector* garmen 13 kelas dilatih dari **penggabungan multi-sumber**: dataset
"Student Dress Code" kampus (Roboflow v3) ditambah **Fashionpedia** dan
**DeepFashion2** untuk memperkaya kelas pelanggaran yang langka (rok mini, celana
sobek, *crop top*, *dress*). Sebuah *unifier* memetakan label tiap sumber ke
kosakata kanonik 13 kelas dan menyeimbangkan kelas langka terlebih dahulu.
Pelatihan dilakukan pada perangkat **CPU** dengan resep hemat (yolo11n @416,
~3 ribu citra, 40 epoch). Evaluasi tetap dilakukan pada *split* validasi/uji
kampus agar tidak bocor oleh data eksternal.

### 3.8 Implementasi

Inti penalaran, CLI, dan API berjalan **tanpa dependensi ML berat** berkat
*backend* persepsi tiruan (*mock*) untuk pengembangan dan pengujian. *Backend* nyata
diaktifkan melalui konfigurasi tunggal `configs/pipeline.yaml`. Tersedia tiga
*backend* yang dapat dipilih: `mock`, `ultralytics` (ringan: orang + garmen),
dan `full` (lengkap: orang → *pose* → parsing Sapiens → garmen → FashionSigLIP).
Layanan **REST API** (FastAPI) memaparkan `/v1/health`, `/v1/profiles`,
`/v1/infer_image`, serta pencatatan kejadian, detail, dan **review/banding** ke
basis data **SQLite**. Antarmuka **web** ("Clothic AI") menyediakan mode unggah
berkas dan webcam dengan pemindaian waktu-nyata. Seluruh artefak model dirujuk
melalui konfigurasi yang menjadi **satu sumber kebenaran**, sehingga API, CLI, dan
skrip verifikasi memuat model yang sama dari berkas yang sama.

---

## 4. HASIL DAN PEMBAHASAN

### 4.1 Verifikasi Fungsional

Seluruh **60 pengujian otomatis lulus** (+1 dilewati), mencakup mesin aturan,
empat skor, fusi temporal, kalibrasi, dan *pipeline* ujung-ke-ujung. Sistem juga
diverifikasi **secara nyata** dengan menjalankan *backend* `full` pada citra asli
melalui REST API yang sama yang dipakai antarmuka web.

Pada citra orang mengenakan atasan tanpa lengan, sistem mengukur keterbukaan
piksel `bahu = 0,97` dan `lengan_atas = 0,95`, mengaktifkan aturan
`UPPER_SLEEVELESS`, menghasilkan **MAJOR_VIOLATION**, beserta langkah perbaikan
terverifikasi ("kenakan atasan berlengan"). Pada citra dengan **celana sobek**,
aturan `RIPPED_CLOTHING` aktif dengan benar; pada citra dengan **atasan *crop*
yang menampakkan perut**, aturan `MIDRIFF_CROP` aktif dengan benar. Hasil ini
menunjukkan bahwa klaim sentral perancangan - **keterbukaan terukur per piksel,
bukan ditebak** - berfungsi dan menghasilkan verdikt yang dapat dijelaskan.

### 4.2 Properti Abstensi

Ketika dijalankan dengan *backend* ringan `ultralytics` (tanpa pengukuran
keterbukaan piksel), sistem dengan tepat memilih `insufficient_evidence` pada citra
yang sama karena `uncertainty` melampaui ambang. Ini mendemonstrasikan perilaku
yang diinginkan: **menolak menuduh** ketika bukti tidak memadai, alih-alih
memaksakan verdikt.

### 4.3 Kinerja dan Batasan Praktis

*Backend* `full` dengan Sapiens 0.3B pada CPU memerlukan ~30–45 detik per citra -
akurat dan cocok untuk **mode unggah berkas**, tetapi belum memadai untuk
pemindaian webcam waktu-nyata tanpa akselerasi GPU/TensorRT. *Backend*
`ultralytics` jauh lebih cepat (~2 detik) namun lebih sering abstain karena tidak
memiliki keterbukaan piksel. *Detector* garmen 13 kelas mencapai
**mAP50 ≈ 0,63** dan **mAP50-95 ≈ 0,43** pada perangkat CPU dengan resep hemat.

### 4.4 Catatan Validitas

Selama verifikasi ditemukan bahwa dua citra contoh yang semula dilabeli secara
informal sebagai "pelanggaran" dan "patuh" sesungguhnya **keduanya merupakan
pelanggaran** (keduanya bercelana sobek; salah satunya *crop top*). Verdikt sistem
karena itu benar; label uji yang diperbaiki. Temuan ini menegaskan perlunya
penyusunan **himpunan uji berlabel yang representatif** - termasuk kasus yang
benar-benar patuh - untuk mengukur secara kuantitatif laju **tuduhan keliru**
(*false-accusation rate*) dan **kesenjangan keadilan** antarkelompok, yang menjadi
agenda evaluasi lanjutan.

---

## 5. ETIKA, KEADILAN, DAN TATA KELOLA

Sistem ini membuat penilaian terhadap tubuh dan pakaian manusia, sehingga keadilan
dan tata kelola diperlakukan sebagai **persyaratan utama**, bukan tambahan.

- **Tanpa model atribut yang dilindungi.** *Pipeline* tidak memuat pengklasifikasi
  gender/etnis/usia. Keputusan hanya bersandar pada geometri garmen + kebijakan.
- **Invarian warna kulit.** Keterbukaan dihitung geometris (parsing + *pose*),
  bukan dari warna kulit; rilis sebaiknya digerbang pada **kesenjangan** metrik
  antar kelompok Fitzpatrick, bukan hanya rata-rata.
- **Pakaian keagamaan diberi daftar-aman.** Hijab, turban, dsb. tidak boleh
  ditandai; perlu disertakan melimpah dalam data dan aturan izin eksplisit.
- **Manusia memutuskan konsekuensi.** Clothic AI hanya mengeluarkan **bendera nasihat**;
  keputusan disipliner tetap di tangan manusia. Tabel `reviews` menegakkan ini.
- **Dapat digugat & transparan.** Setiap keputusan menyertakan bukti + kutipan
  sehingga mahasiswa dapat membanding terhadap pengukuran dan aturan yang berlaku;
  kebijakan JSON dapat dipublikasikan.
- **Privasi & minimisasi data.** Mengutamakan penyimpanan vektor bukti dibanding
  bingkai mentah; enkripsi, retensi singkat, dan penghapusan otomatis.

---

## 6. DAMPAK

**Dampak akademik.** Clothic AI menunjukkan pola perancangan yang dapat
direplikasi: **memisahkan persepsi (objektif, dipelajari) dari penilaian normatif
(transparan, dikonfigurasi)**. Pola ini relevan jauh melampaui tata busana -
berlaku untuk setiap sistem CV yang harus memutuskan hal normatif (keselamatan
kerja, kelayakan APD, moderasi). Sifatnya yang dapat dijelaskan, dapat digugat, dan
adil menjadikannya **dapat dipertahankan sebagai karya ilmiah/skripsi**, bukan
sekadar demo *black box*.

**Dampak institusional.** Kampus memperoleh penegakan yang **konsisten dan dapat
diaudit**: aturan menjadi eksplisit (JSON + kutipan), setiap keputusan tercatat,
dan terdapat jalur banding formal. Mengubah kebijakan cukup menyunting berkas
konfigurasi - **tanpa pelatihan ulang model dan tanpa perubahan kode** - sehingga
biaya pemeliharaan rendah dan adaptasi antarprofil (ujian, lab) cepat.

**Dampak sosial dan etis.** Dengan menolak menebak di bawah ketidakpastian dan
dengan menghapus model atribut yang dilindungi, Clothic AI mengurangi risiko
**tuduhan keliru** dan **bias** yang melekat pada pengklasifikasi biner. Keputusan
yang menyertakan bukti dan kutipan memberi mahasiswa **martabat untuk memahami dan
menggugat** - mengubah pengawasan yang berpotensi menindas menjadi alat bantu
kebijakan yang **akuntabel dan legitimat**.

**Dampak teknis/rekayasa.** Arsitektur modular bertipe memungkinkan tiap komponen
(detektor, parser, pengklasifikasi atribut) ditingkatkan secara independen;
konfigurasi tunggal sebagai sumber kebenaran menyederhanakan operasi; dan jalur
*deployment* (kuantisasi, TensorRT/Triton) telah dirancang untuk transisi ke
waktu-nyata.

---

## 7. KESIMPULAN

Penelitian ini merancang dan mengimplementasikan **Clothic AI**, sistem pemantauan
kepatuhan berpakaian kampus yang **dapat dijelaskan**. Dengan memisahkan persepsi
visual objektif dari penalaran kebijakan yang transparan, Clothic AI menghindari
kelemahan fundamental pengklasifikasi biner "sopan/tidak sopan": ia tidak pernah
mempelajari penilaian subjektif, melainkan mengukur atribut yang dapat
dipertanggungjawabkan (terutama keterbukaan **berbasis piksel** dari *human
parsing*) dan menyerahkan keputusan kepada mesin aturan JSON yang dapat disunting.
Empat skor terkalibrasi dan mekanisme **abstensi** menjadikan sistem lebih adil dan
andal, sementara penjelasan beserta *counterfactual* membuatnya dapat digugat.
Sistem telah diverifikasi secara fungsional (60 pengujian lulus) dan secara nyata
pada citra asli. Dampak utamanya adalah transformasi pengawasan kabur menjadi alat
bantu kebijakan yang **akuntabel, dapat dikonfigurasi, dan menghormati keadilan**.

---

## 8. KETERBATASAN DAN PENGEMBANGAN LANJUT

1. **Latensi waktu-nyata.** Sapiens 0.3B pada CPU ~30–45 dtk/citra; perlu
   GPU/kuantisasi/TensorRT untuk pemindaian webcam waktu-nyata.
2. **Evaluasi kuantitatif.** Diperlukan himpunan uji berlabel yang representatif
   (termasuk kasus benar-benar patuh dan beragam kelompok Fitzpatrick) untuk
   mengukur presisi/recall, laju tuduhan keliru, dan kesenjangan keadilan.
3. **Atribut masih *zero-shot*.** Pengklasifikasi atribut FashionSigLIP belum
   dilatih *probe*-nya; kalibrasi *prompt* (mis. suhu *softmax*) perlu disetel pada
   data berlabel agar probabilitas tidak terlalu yakin.
4. **Penyempurnaan parsing & pose.** *Fine-tune* parsing (LIP/ATR/CIHP) dan opsi
   RTMPose untuk anchor region yang lebih akurat.
5. **Tata kelola produksi.** Pemantauan bias berkelanjutan, mode bayangan
   (*shadow*), dan persetujuan etik untuk penangkapan data mahasiswa.

---

## DAFTAR PUSTAKA (RUJUKAN UTAMA)

1. Meta AI. *Sapiens: Foundation for Human Vision Models* (Goliath 28-class body
   parsing). https://huggingface.co/facebook/sapiens
2. Zhai, X., dkk. *Sigmoid Loss for Language Image Pre-Training (SigLIP)* /
   *SigLIP 2*. arXiv:2502.14786.
3. Marqo AI. *Marqo-FashionSigLIP / FashionCLIP*.
   https://github.com/marqo-ai/marqo-FashionCLIP
4. Jocher, G., dkk. *Ultralytics YOLO* (deteksi & pose).
5. Zhang, Y., dkk. *ByteTrack: Multi-Object Tracking by Associating Every
   Detection Box*. ECCV 2022.
6. Guo, C., dkk. *On Calibration of Modern Neural Networks* (temperature scaling,
   ECE). ICML 2017.
7. Ge, Y., dkk. *DeepFashion2: A Versatile Benchmark for Detection, Pose
   Estimation, Segmentation and Re-Identification of Clothing Images*. CVPR 2019.
8. Jia, M., dkk. *Fashionpedia: Ontology, Segmentation, and an Attribute
   Localization Dataset*. ECCV 2020.

---

*Dokumen ini merangkum spesifikasi lengkap pada `docs/ARCHITECTURE.md`,
pemilihan model pada `docs/MODELS.md`, dan status implementasi pada
`docs/STATUS.md`.*
