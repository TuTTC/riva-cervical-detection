import sys
import os
import torch

# Đường dẫn tới thư mục chứa code mô hình
FRCNN_DIR = r"d:\NĂM 3\RIVA\RIVA-TRACK-A\frcnn_paper"
sys.path.append(FRCNN_DIR)

try:
    from model import create_model
    from config import IMG_SIZE
except ImportError as e:
    print(f"Lỗi không thể import model: {e}")
    print("Vui lòng kiểm tra lại đường dẫn tới FRCNN_DIR.")
    sys.exit(1)

def export_frcnn_to_onnx():
    weights_path = r"d:\NĂM 3\RIVA\models\faster_rcnn_final.pth"
    onnx_path = r"d:\NĂM 3\RIVA\models\faster_rcnn_final.onnx"
    
    device = torch.device('cpu') # Export trên CPU cho an toàn và tránh lỗi memory
    print(f"1. Đang tải model Faster R-CNN từ {weights_path}...")
    
    # Khởi tạo kiến trúc model
    model = create_model(pretrained=False)
    
    # Tải trọng số
    checkpoint = torch.load(weights_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    print("   ✓ Đã tải xong trọng số.")
    
    # 2. Tạo input giả định (Dummy Input)
    # Model torchvision nhận input shape [Batch, Channels, Height, Width]
    # Dùng kích thước mặc định từ config (ví dụ 1200x1200)
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    
    print(f"2. Bắt đầu xuất ra định dạng ONNX tại {onnx_path}...")
    
    # 3. Thực hiện Export
    # Opset 11 là điều kiện tối thiểu để hỗ trợ RoI Align và NMS của Faster R-CNN
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11, 
            do_constant_folding=True,
            input_names=['input_images'],
            output_names=['boxes', 'labels', 'scores'], # Torchvision trả ra 3 output này
            dynamic_axes={
                'input_images': {0: 'batch_size', 2: 'height', 3: 'width'}, # Hỗ trợ ảnh kích thước khác nhau
                'boxes': {0: 'batch_size', 1: 'num_boxes'},
                'labels': {0: 'batch_size', 1: 'num_boxes'},
                'scores': {0: 'batch_size', 1: 'num_boxes'}
            }
        )
        print("   ✓ Xuất ONNX thành công!")
        print(f"   => File ONNX đã được lưu tại: {onnx_path}")
    except Exception as e:
        print(f"Xảy ra lỗi trong quá trình xuất ONNX: {e}")

if __name__ == "__main__":
    export_frcnn_to_onnx()
