# OCR Tester — Modular Edition

## Project Structure

```
ocr_tester/
├── ocr_tester.py          ← Entry point (CLI)
├── config/
│   ├── __init__.py
│   └── settings.py        ← All constants, engine registry, defaults
├── engines/
│   ├── __init__.py
│   ├── base.py            ← OCRResult type + base interface
│   ├── paddle_engine.py   ← PaddleOCR PP-OCRv5
│   ├── doctr_engine.py    ← docTR (db_resnet50 + parseq)
│   ├── rapidocr_engine.py ← RapidOCR ONNX (CPU)
│   ├── rapidocr_paddle_engine.py ← RapidOCR Paddle (GPU)
│   ├── easyocr_engine.py  ← EasyOCR
│   └── tesseract_engine.py← Tesseract 5.x
├── metrics/
│   ├── __init__.py
│   ├── confidence.py      ← Metric 1: confidence distribution
│   ├── cer.py             ← Metric 2: character error rate
│   └── field_check.py     ← Metric 3: known-field spot check
├── utils/
│   ├── __init__.py
│   ├── cuda_fix.py        ← Windows CUDA DLL path fix
│   ├── pdf_utils.py       ← PDF rendering + text layer extraction
│   ├── device_probe.py    ← GPU/CPU evidence collection
│   ├── known_fields.py    ← known_fields.txt loader
│   ├── flagging.py        ← Critical pattern flagging
│   ├── output_writer.py   ← Per-engine report writer
│   └── comparison.py      ← Cross-engine comparison table
└── input_files/
    ├── your_scan.pdf
    └── known_fields.txt   ← optional; enables Metric 3
```

## Known Issues & Fixes

### 1. PaddlePaddle CUDA DLL not found (`cudnn_cnn64_9.dll`)
`os.add_dll_directory` does NOT work for paddle's internal DLL loader on Windows.

**Fix — run once after installing:**
```cmd
copy "venv\Lib\site-packages\nvidia\cudnn\bin\*.dll"        "venv\Lib\site-packages\paddle\"
copy "venv\Lib\site-packages\nvidia\cublas\bin\*.dll"       "venv\Lib\site-packages\paddle\"
copy "venv\Lib\site-packages\nvidia\cuda_runtime\bin\*.dll" "venv\Lib\site-packages\paddle\"
```
Verify: `python -c "import paddle; print(paddle.device.get_device())"` → should print `gpu:0`

### 2. torchvision still on CPU build
```cmd
pip uninstall torchvision -y
pip install torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
```

### 3. Duplicate OpenCV (3 variants installed)
```cmd
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless==4.11.0.86
```

### 4. Tesseract binary not in PATH
Download from: https://github.com/UB-Mannheim/tesseract/wiki
Add install folder to system PATH, then verify:
```cmd
tesseract --version
```

### 5. Paddle module state corrupted by early import
The `get_device_evidence()` call used to import paddle at startup; if the DLL
fix (issue #1) has not been applied yet, this corrupts the paddle module in
`sys.modules` and causes `rapidocr_paddle` to fail too.

**Fix in code:** `device_probe.py` now wraps every engine import in an
isolated try/except and never imports paddle unless it can be cleanly loaded.
`run_rapidocr_paddle()` also purges any stale `paddle*` entries from
`sys.modules` before attempting its own import.

## Usage

```bash
# Activate venv first
cd "C:\Public Bank\ocr_testing"
venv\Scripts\activate

# Single engine, CPU
python ocr_tester.py --folder input_files --engines paddle

# Single engine, GPU
python ocr_tester.py --folder input_files --engines paddle --gpu

# All engines, GPU, higher DPI for degraded scans
python ocr_tester.py --folder input_files --engines all --gpu --dpi 400

# Compare two specific engines
python ocr_tester.py --folder input_files --engines rapidocr rapidocr_paddle --gpu
```
