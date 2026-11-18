import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

os.makedirs("static/samples", exist_ok=True)

def create_endoscopic_frame(filename, polyp_type="sessile"):
    """
    Synthesize realistic endoscopic colonoscopy mucosal frame and paired binary mask.
    """
    w, h = 512, 512
    # 1. Base mucosal color (pinkish-red fleshy colon wall)
    # Radial falloff (dark periphery due to endoscope circular vignette/lumen)
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_r = np.sqrt(cx**2 + cy**2)
    vignette = 1.0 - (r / (max_r * 0.95))**1.8
    vignette = np.clip(vignette, 0.1, 1.0)

    # Base mucosal tissue texture
    np.random.seed(42 if polyp_type == "sessile" else (101 if polyp_type == "pedunculated" else 777))
    noise = np.random.normal(0, 8, (h, w))
    
    # Base colon tone: deep pinkish red (R~185, G~75, B~65)
    base_r = np.clip(180 + noise * 1.5 - (r / max_r) * 60, 20, 255)
    base_g = np.clip(75 + noise * 1.0 - (r / max_r) * 45, 10, 200)
    base_b = np.clip(70 + noise * 0.8 - (r / max_r) * 40, 10, 190)

    img_arr = np.stack([base_r * vignette, base_g * vignette, base_b * vignette], axis=-1).astype(np.uint8)
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)

    # 2. Add subtle submucosal micro-vascular blood vessels
    for _ in range(12):
        vx = np.random.randint(50, 450)
        vy = np.random.randint(50, 450)
        points = [(vx, vy)]
        for _ in range(6):
            vx += np.random.randint(-25, 25)
            vy += np.random.randint(-25, 25)
            points.append((vx, vy))
        draw.line(points, fill=(130, 30, 25, 160), width=np.random.randint(1, 3))

    # Mask array
    mask = Image.new("L", (w, h), 0)
    mask_draw = ImageDraw.Draw(mask)

    if polyp_type == "sessile":
        # Elevated fleshy hemispherical lesion around (280, 240)
        px, py, prx, pry = 280, 240, 75, 65
        # Draw lesion mask
        mask_draw.ellipse([px - prx, py - pry, px + prx, py + pry], fill=255)
        # Extra lobule
        mask_draw.ellipse([px + 20 - 45, py - 30 - 35, px + 20 + 45, py - 30 + 35], fill=255)
        
        # Draw lesion in image with reddish-hyperemic texture and glandular surface
        lesion_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        l_draw = ImageDraw.Draw(lesion_overlay)
        l_draw.ellipse([px - prx, py - pry, px + prx, py + pry], fill=(215, 65, 75, 230))
        l_draw.ellipse([px + 20 - 45, py - 30 - 35, px + 20 + 45, py - 30 + 35], fill=(225, 75, 80, 230))
        
        # Specular light reflections (moist mucosal sheen)
        l_draw.ellipse([px - 25, py - 30, px - 10, py - 20], fill=(255, 255, 255, 210))
        l_draw.ellipse([px + 15, py - 40, px + 25, py - 32], fill=(255, 255, 255, 180))
        l_draw.line([(px - 30, py), (px + 30, py + 10)], fill=(160, 20, 30, 180), width=2)
        
        lesion_overlay = lesion_overlay.filter(ImageFilter.GaussianBlur(radius=2))
        img = Image.alpha_composite(img.convert("RGBA"), lesion_overlay).convert("RGB")

    elif polyp_type == "pedunculated":
        # Polyp on a stalk at (220, 260) with stalk from (150, 350)
        px, py, prx, pry = 230, 230, 60, 55
        sx, sy = 160, 320
        # Stalk
        mask_draw.polygon([(sx - 15, sy), (px - 20, py + 30), (px + 15, py + 30), (sx + 15, sy)], fill=255)
        # Head
        mask_draw.ellipse([px - prx, py - pry, px + prx, py + pry], fill=255)

        lesion_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        l_draw = ImageDraw.Draw(lesion_overlay)
        # Stalk
        l_draw.polygon([(sx - 15, sy), (px - 20, py + 30), (px + 15, py + 30), (sx + 15, sy)], fill=(190, 60, 65, 240))
        # Head
        l_draw.ellipse([px - prx, py - pry, px + prx, py + pry], fill=(230, 55, 65, 250))
        # Specular reflections
        l_draw.ellipse([px - 20, py - 20, px - 5, py - 10], fill=(255, 255, 255, 230))
        l_draw.ellipse([px + 10, py - 25, px + 22, py - 18], fill=(255, 255, 255, 200))
        
        lesion_overlay = lesion_overlay.filter(ImageFilter.GaussianBlur(radius=2))
        img = Image.alpha_composite(img.convert("RGBA"), lesion_overlay).convert("RGB")

    elif polyp_type == "flat":
        # Paris IIa flat subtle lesion at (260, 260)
        px, py, prx, pry = 260, 260, 90, 70
        mask_draw.ellipse([px - prx, py - pry, px + prx, py + pry], fill=255)
        
        lesion_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        l_draw = ImageDraw.Draw(lesion_overlay)
        l_draw.ellipse([px - prx, py - pry, px + prx, py + pry], fill=(205, 90, 95, 140))
        l_draw.ellipse([px - 10, py - 15, px + 5, py - 5], fill=(255, 245, 240, 160))
        lesion_overlay = lesion_overlay.filter(ImageFilter.GaussianBlur(radius=4))
        img = Image.alpha_composite(img.convert("RGBA"), lesion_overlay).convert("RGB")

    # Final organic polish & moisture glints
    draw_final = ImageDraw.Draw(img)
    # Endoscopic circular scope border / aperture
    draw_final.ellipse([10, 10, w - 10, h - 10], outline=(15, 10, 10), width=8)

    # Save files
    img.save(f"static/samples/{filename}.jpg", quality=95)
    mask.save(f"static/samples/{filename}_mask.png")
    print(f"Generated {filename}.jpg and paired mask.")

create_endoscopic_frame("sample_sessile", "sessile")
create_endoscopic_frame("sample_pedunculated", "pedunculated")
create_endoscopic_frame("sample_flat_adenoma", "flat")
create_endoscopic_frame("sample_normal_mucosa", "normal")
