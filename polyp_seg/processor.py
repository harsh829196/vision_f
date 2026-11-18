import io
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

def pil_to_base64(img, format="PNG", quality=95):
    """Convert PIL Image to base64 data URI."""
    buffered = io.BytesIO()
    if format.upper() == "JPEG":
        img = img.convert("RGB")
        img.save(buffered, format=format, quality=quality)
        mime = "image/jpeg"
    else:
        img.save(buffered, format=format)
        mime = "image/png"
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def base64_to_pil(b64_string):
    """Convert base64 data URI to PIL Image."""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def colormap_jet(prob_map):
    """
    Generate Jet / Turbo style RGB heatmap from 2D normalized float probability array [0, 1].
    Implemented with pure NumPy for zero OpenCV dependency!
    """
    val = np.clip(prob_map, 0.0, 1.0)
    
    # 4-stage color ramps (Blue -> Cyan -> Green -> Yellow -> Red)
    r = np.clip(1.5 - np.abs(val * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(val * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(val * 4.0 - 1.0), 0.0, 1.0)

    heatmap_np = np.stack([r * 255, g * 255, b * 255], axis=-1).astype(np.uint8)
    return Image.fromarray(heatmap_np)


def create_overlay(orig_img, mask_np, color_rgb=(16, 185, 129), opacity=0.55):
    """
    Generate aesthetic semi-transparent color blend overlay of segmented lesion onto original frame.
    color_rgb: RGB tuple, e.g. (16, 185, 129) for emerald green, (239, 68, 68) for red, etc.
    """
    orig_rgba = orig_img.convert("RGBA")
    w, h = orig_img.size

    # Ensure mask_np is [H, W] uint8
    if mask_np.shape != (h, w):
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
        mask_np = np.array(mask_pil) > 127
    else:
        mask_np = mask_np > 0.5

    color_layer = np.zeros((h, w, 4), dtype=np.uint8)
    alpha_val = int(255 * np.clip(opacity, 0.0, 1.0))

    color_layer[mask_np, 0] = color_rgb[0]
    color_layer[mask_np, 1] = color_rgb[1]
    color_layer[mask_np, 2] = color_rgb[2]
    color_layer[mask_np, 3] = alpha_val

    overlay_img = Image.fromarray(color_layer, mode="RGBA")
    composite = Image.alpha_composite(orig_rgba, overlay_img)
    return composite.convert("RGB")


def extract_lesion_analysis(orig_img, prob_map, threshold=0.45):
    """
    Analyze probability map and extract lesion metrics:
    - Detection boolean & confidence
    - Lesion area percentage & pixel count
    - Bounding box (x, y, w, h) & Centroid (cx, cy)
    - Paris classification heuristic
    - Delineated contour overlay
    """
    w, h = orig_img.size
    prob_resized = np.array(Image.fromarray((prob_map * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    binary_mask = (prob_resized >= threshold).astype(np.uint8)

    lesion_pixels = int(np.sum(binary_mask))
    total_pixels = w * h
    area_pct = round((lesion_pixels / total_pixels) * 100, 2)
    detected = lesion_pixels > 200

    # Calculate confidence
    if detected:
        mean_conf = float(np.mean(prob_resized[binary_mask == 1]))
        max_conf = float(np.max(prob_resized))
        confidence = round(float((mean_conf * 0.4 + max_conf * 0.6) * 100), 1)
        confidence = min(max(confidence, 65.0), 99.4)
    else:
        max_conf = float(np.max(prob_resized)) if prob_resized.size > 0 else 0.0
        confidence = round(float(max_conf * 100), 1)
        confidence = min(confidence, 32.0)

    # Compute bounding box and centroid if detected
    bbox = None
    centroid = None
    paris_class = "Normal (No Lesion Detected)"
    morphology = "Clean mucosal tissue with uniform vascular architecture."
    risk_level = "Tier 0: Negative / Benign Mucosa"
    recommendation = "Standard routine surveillance per GI screening protocol."

    contour_img = orig_img.copy()
    draw = ImageDraw.Draw(contour_img)

    if detected:
        y_indices, x_indices = np.where(binary_mask == 1)
        x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
        y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
        bbox = {
            "x": x_min,
            "y": y_min,
            "w": x_max - x_min + 1,
            "h": y_max - y_min + 1
        }
        centroid = {
            "x": int(np.mean(x_indices)),
            "y": int(np.mean(y_indices))
        }

        # Paris Classification heuristic
        aspect_ratio = bbox["w"] / max(bbox["h"], 1)
        est_diameter_px = max(bbox["w"], bbox["h"])

        if aspect_ratio > 1.6 or aspect_ratio < 0.6:
            paris_class = "Type Ip (Pedunculated / Stalked)"
            morphology = "Prominent polyp head supported by narrow vascularized stalk."
            risk_level = "Tier 2: Neoplastic Adenoma Suspect - Polypectomy Indicated"
            recommendation = "Endoscopic snare polypectomy with histopathological margin evaluation."
        elif area_pct > 6.0:
            paris_class = "Type Is (Sessile Adenoma)"
            morphology = "Broad-based elevated lesion with glandular mucosal distortion."
            risk_level = "Tier 3: Elevated Sessile Adenoma - Resection Recommended"
            recommendation = "Endoscopic Mucosal Resection (EMR) or submucosal dissection assessment."
        else:
            paris_class = "Type IIa (Slightly Elevated / Flat)"
            morphology = "Flat mucosal discoloration with subtle border elevation."
            risk_level = "Tier 1: Subtle Flat Lesion - Narrow-Band Imaging (NBI) Advised"
            recommendation = "Optical chromoendoscopy enhancement to delineate lateral margins."

        # Draw glowing contour boundary & bounding box
        # Border contour using edge filter on binary mask
        mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
        edges = mask_pil.filter(ImageFilter.FIND_EDGES)
        edge_np = np.array(edges) > 50

        contour_overlay = np.zeros((h, w, 4), dtype=np.uint8)
        # Laser Cyan / Emerald edge glow
        contour_overlay[edge_np, 0] = 6
        contour_overlay[edge_np, 1] = 230
        contour_overlay[edge_np, 2] = 180
        contour_overlay[edge_np, 3] = 255

        contour_pil = Image.fromarray(contour_overlay, mode="RGBA")
        contour_img = Image.alpha_composite(contour_img.convert("RGBA"), contour_pil).convert("RGB")
        draw_final = ImageDraw.Draw(contour_img)

        # Draw tech bounding box brackets with corner accents
        pad = 6
        bx1 = max(0, x_min - pad)
        by1 = max(0, y_min - pad)
        bx2 = min(w - 1, x_max + pad)
        by2 = min(h - 1, y_max + pad)
        corner_len = min(20, (bx2 - bx1) // 4, (by2 - by1) // 4)

        # Tech corner accents
        c_color = (6, 230, 180)
        # Top-left
        draw_final.line([(bx1, by1), (bx1 + corner_len, by1)], fill=c_color, width=3)
        draw_final.line([(bx1, by1), (bx1, by1 + corner_len)], fill=c_color, width=3)
        # Top-right
        draw_final.line([(bx2, by1), (bx2 - corner_len, by1)], fill=c_color, width=3)
        draw_final.line([(bx2, by1), (bx2, by1 + corner_len)], fill=c_color, width=3)
        # Bottom-left
        draw_final.line([(bx1, by2), (bx1 + corner_len, by2)], fill=c_color, width=3)
        draw_final.line([(bx1, by2), (bx1, by2 - corner_len)], fill=c_color, width=3)
        # Bottom-right
        draw_final.line([(bx2, by2), (bx2 - corner_len, by2)], fill=c_color, width=3)
        draw_final.line([(bx2, by2), (bx2, by2 - corner_len)], fill=c_color, width=3)

        # Target centroid crosshair
        cx, cy = centroid["x"], centroid["y"]
        draw_final.line([(cx - 7, cy), (cx + 7, cy)], fill=(255, 60, 60), width=2)
        draw_final.line([(cx, cy - 7), (cx, cy + 7)], fill=(255, 60, 60), width=2)

    # Convert binary mask to displayable RGB image
    mask_rgb = np.stack([binary_mask * 255, binary_mask * 255, binary_mask * 255], axis=-1).astype(np.uint8)
    mask_display = Image.fromarray(mask_rgb)

    return {
        "detected": detected,
        "confidence": confidence,
        "area_pct": area_pct,
        "lesion_pixels": lesion_pixels,
        "total_pixels": total_pixels,
        "paris_class": paris_class,
        "morphology": morphology,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "bbox": bbox,
        "centroid": centroid,
        "mask_img": mask_display,
        "contour_img": contour_img,
        "binary_mask_np": binary_mask
    }

