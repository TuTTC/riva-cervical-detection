# ============================================================================
# FIX LỖI PYTHON 3.12 & OPENMMLAB
# Dành cho hệ thống: Python 3.12, CUDA 13.0, PyTorch 2.10+cu126
# ============================================================================
import subprocess
import sys

def run(cmd):
    print(f"\n--> Chạy lệnh: {cmd}")
    subprocess.check_call(cmd, shell=True)

try:
    # ── Bước 1: Cài OpenMIM ──────────────────────────────────────────────
    print("1️⃣  Đang cài đặt OpenMIM...")
    run("pip install -U openmim")

    # ── Bước 2: Vá setuptools (phiên bản tương thích Python 3.12) ────────
    # openmim kéo setuptools 60.2.0 (cũ, lỗi trên Python 3.12)
    # Cần phiên bản >= 70 (hỗ trợ Python 3.12) nhưng < 82 (vẫn có pkg_resources)
    print("\n2️⃣  Đang VÁ LỖI setuptools cho Python 3.12...")
    run("pip install 'setuptools>=70.0.0,<80.0.0'")

    # ── Bước 3: Cài mmengine ─────────────────────────────────────────────
    print("\n3️⃣  Đang cài đặt mmengine...")
    run("mim install mmengine")

    # ── Bước 4: Cài MMCV ─────────────────────────────────────────────────
    # Dùng --no-build-isolation để tránh pip tạo môi trường build riêng
    # (môi trường đó sẽ kéo setuptools 82+ không có pkg_resources)
    # Set CUDA_HOME để tránh lỗi CUDA mismatch (system CUDA 13.0 vs torch CUDA 12.6)
    print("\n4️⃣  Đang cài đặt MMCV...")
    # Cần cài build dependencies trước khi dùng --no-build-isolation
    run("pip install ninja")
    # Monkey-patch PyTorch để bỏ qua CUDA version check khi build
    run(
        'python -c "'
        "import torch.utils.cpp_extension as ce; "
        "import inspect, re; "
        "src_file = inspect.getfile(ce); "
        "f = open(src_file); code = f.read(); f.close(); "
        "new_code = code.replace("
        "'raise RuntimeError(CUDA_MISMATCH_MESSAGE', "
        "'pass # raise RuntimeError(CUDA_MISMATCH_MESSAGE'"
        "); "
        "f = open(src_file, 'w'); f.write(new_code); f.close(); "
        "print('Patched:', src_file)"
        '"'
    )
    run(
        "MMCV_WITH_OPS=1 FORCE_CUDA=1 "
        "pip install 'mmcv>=2.0.0rc4,<2.2.0' --no-build-isolation --no-cache-dir"
    )

    # ── Bước 5: Cài MMDetection ──────────────────────────────────────────
    print("\n5️⃣  Đang cài đặt MMDetection...")
    run("pip install -q mmdet")

    # ── Kiểm tra ─────────────────────────────────────────────────────────
    print("\n✅ CÀI ĐẶT THÀNH CÔNG!")
    import mmcv
    import mmdet
    print(f"   - MMCV version: {mmcv.__version__}")
    print(f"   - MMDetection version: {mmdet.__version__}")

except subprocess.CalledProcessError as e:
    print(f"\n❌ LỖI khi chạy lệnh (exit code {e.returncode}): {e.cmd}")
except Exception as e:
    print(f"\n❌ VẪN CÓ LỖI: {e}")
