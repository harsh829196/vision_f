import os
import io
import json
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
from polyp_seg.predictor import PolypPredictor

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload

# Initialize predictor
predictor = PolypPredictor(checkpoint_path="runs/vgg16_unet/best.pth")

# Predefined samples metadata
SAMPLE_REGISTRY = {
    "sessile": {
        "id": "sessile",
        "title": "Sessile Adenoma (Paris Type Is)",
        "subtitle": "Broad-based elevated mucosal polyp with crypt distortion",
        "image_file": "sample_sessile.jpg",
        "mask_file": "sample_sessile_mask.png",
        "paris_class": "Type Is",
        "clinical_note": "Elevated sessile lesion in the sigmoid colon. Margins clearly delineated."
    },
    "pedunculated": {
        "id": "pedunculated",
        "title": "Pedunculated Polyp (Paris Type Ip)",
        "subtitle": "Polypoid head with vascularized supporting stalk",
        "image_file": "sample_pedunculated.jpg",
        "mask_file": "sample_pedunculated_mask.png",
        "paris_class": "Type Ip",
        "clinical_note": "Descending colon stalked lesion with active mucosal vascularization."
    },
    "flat": {
        "id": "flat",
        "title": "Flat Mucosal Lesion (Paris Type IIa)",
        "subtitle": "Subtle, slightly elevated superficial dysplastic area",
        "image_file": "sample_flat_adenoma.jpg",
        "mask_file": "sample_flat_adenoma_mask.png",
        "paris_class": "Type IIa",
        "clinical_note": "Ascending colon flat adenoma, difficult to detect with white light alone."
    },
    "normal": {
        "id": "normal",
        "title": "Normal Mucosa (Control / Negative)",
        "subtitle": "Healthy colonic mucosa without neoplastic lesions",
        "image_file": "sample_normal_mucosa.jpg",
        "mask_file": "sample_normal_mucosa_mask.png",
        "paris_class": "None",
        "clinical_note": "Normal mucosal pattern and regular branching submucosal venules."
    }
}

def hex_to_rgb(hex_code):
    """Convert hex color (#10b981 or 10b981) to RGB tuple."""
    hex_code = hex_code.lstrip("#")
    if len(hex_code) == 6:
        return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    return (16, 185, 129)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Return available sample cases."""
    samples_list = []
    for s_id, s_data in SAMPLE_REGISTRY.items():
        samples_list.append({
            "id": s_id,
            "title": s_data["title"],
            "subtitle": s_data["subtitle"],
            "paris_class": s_data["paris_class"],
            "clinical_note": s_data["clinical_note"],
            "image_url": f"/static/samples/{s_data['image_file']}"
        })
    return jsonify({"status": "success", "samples": samples_list})


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        sample_id = request.form.get("sample_id") or request.args.get("sample_id")
        color_hex = request.form.get("color", "#10b981")
        opacity = float(request.form.get("opacity", 0.55))
        threshold = float(request.form.get("threshold", 0.45))
        color_rgb = hex_to_rgb(color_hex)

        img_pil = None
        paired_mask_pil = None

        # 1. Handle Preset Sample
        if sample_id and sample_id in SAMPLE_REGISTRY:
            s_data = SAMPLE_REGISTRY[sample_id]
            img_path = os.path.join(app.static_folder, "samples", s_data["image_file"])
            mask_path = os.path.join(app.static_folder, "samples", s_data["mask_file"])
            if os.path.exists(img_path):
                img_pil = Image.open(img_path).convert("RGB")
            if os.path.exists(mask_path):
                paired_mask_pil = Image.open(mask_path).convert("L")

        # 2. Handle Uploaded File
        elif "file" in request.files:
            file = request.files["file"]
            if file and file.filename != "":
                img_pil = Image.open(file.stream).convert("RGB")

        # 3. Handle JSON payload (base64 image)
        elif request.is_json:
            data = request.get_json()
            if "image" in data:
                b64 = data["image"]
                if "," in b64:
                    b64 = b64.split(",")[1]
                import base64
                img_data = base64.b64decode(b64)
                img_pil = Image.open(io.BytesIO(img_data)).convert("RGB")
            if "color" in data:
                color_rgb = hex_to_rgb(data["color"])
            if "opacity" in data:
                opacity = float(data["opacity"])
            if "threshold" in data:
                threshold = float(data["threshold"])

        if img_pil is None:
            return jsonify({"status": "error", "message": "No valid image provided"}), 400

        # Run inference
        result = predictor.predict(
            img_pil=img_pil,
            paired_mask_pil=paired_mask_pil,
            color_rgb=color_rgb,
            opacity=opacity,
            threshold=threshold
        )

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Return model specifications and benchmark training metrics."""
    return jsonify({
        "status": "success",
        "name": "EndoVision-VGG16-UNet",
        "description": "Deep Convolutional Neural Network for Endoscopic Polyp and GI Lesion Semantic Segmentation",
        "architecture": "U-Net Encoder-Decoder with Skip Connections",
        "encoder": "VGG-16 (ImageNet-1k pre-trained feature extractor)",
        "loss_function": "Combined Binary Cross-Entropy + Soft Dice Loss",
        "dataset": "Kvasir-SEG (Simula Research Laboratory, 1000 annotated colonoscopy frames)",
        "metrics": {
            "mean_dice": 0.8915,
            "polyp_iou": 0.8301,
            "mean_iou": 0.8931,
            "val_loss": 0.4941,
            "epochs_trained": 40
        },
        "target_lesions": [
            "Sessile Serrated Lesions (SSL)",
            "Tubular & Tubulovillous Adenomas",
            "Hyperplastic Colorectal Polyps",
            "Flat Dysplastic Mucosa (Paris Type IIa/b)"
        ],
        "device": str(predictor.device).upper(),
        "disclaimer": "For research, educational and diagnostic assistance demonstration only. Not cleared as an autonomous diagnostic medical device."
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print("=" * 60)
    print(" EndoVision AI - Endoscopic Polyp & Lesion Segmentation")
    print(f" Starting Flask Server at http://127.0.0.1:{port}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=True)