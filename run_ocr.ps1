# run_ocr.ps1
# -----------
# Use this instead of `python ocr_tester.py` directly.
#
# These flags MUST be set as env vars before Python starts.
# Setting them via os.environ inside Python is too late —
# paddle reads them at C++ runtime init on the first `import paddle`.

$env:FLAGS_use_mkldnn                  = "0"
$env:FLAGS_enable_pir_api              = "0"
$env:FLAGS_use_new_executor            = "0"
$env:PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT = "0"

python ocr_tester.py @args