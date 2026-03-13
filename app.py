import os
import tempfile
import zipfile
from functools import lru_cache

import gradio as gr
import numpy as np
from rembg import remove, new_session
from PIL import Image


@lru_cache(maxsize=4)
def get_session(model_name):
    """Cache rembg sessions so the model is only loaded once per model name."""
    return new_session(model_name)

# Common Pantone colors mapped to RGB values
PANTONE_COLORS = {
    "": None,
    "100 C (Yellow)": (244, 237, 71),
    "109 C (Golden Yellow)": (255, 209, 0),
    "116 C (Bright Yellow)": (255, 205, 0),
    "151 C (Orange)": (255, 130, 0),
    "172 C (Red Orange)": (250, 70, 22),
    "185 C (Red)": (228, 0, 43),
    "186 C (True Red)": (200, 16, 46),
    "199 C (Dark Red)": (161, 35, 56),
    "210 C (Pink)": (255, 121, 176),
    "219 C (Hot Pink)": (218, 24, 132),
    "253 C (Purple)": (165, 51, 176),
    "267 C (Violet)": (89, 44, 164),
    "280 C (Navy)": (1, 33, 105),
    "286 C (Blue)": (0, 51, 160),
    "293 C (Royal Blue)": (0, 56, 168),
    "299 C (Cyan Blue)": (0, 163, 224),
    "306 C (Light Blue)": (0, 181, 226),
    "320 C (Teal)": (0, 133, 125),
    "333 C (Aqua Green)": (0, 155, 119),
    "347 C (Green)": (0, 154, 68),
    "355 C (Bright Green)": (0, 177, 64),
    "375 C (Lime Green)": (151, 215, 0),
    "382 C (Yellow Green)": (196, 214, 0),
    "401 C (Warm Gray)": (175, 169, 160),
    "418 C (Cool Gray)": (130, 137, 133),
    "425 C (Charcoal)": (84, 88, 90),
    "432 C (Dark Slate)": (51, 63, 72),
    "448 C (Dark Olive)": (74, 65, 42),
    "476 C (Dark Brown)": (89, 60, 31),
    "485 C (Bright Red)": (218, 41, 28),
    "Black C": (45, 41, 38),
    "White": (255, 255, 255),
    "Cool Gray 1 C": (217, 214, 209),
    "Cool Gray 5 C": (177, 179, 179),
    "Cool Gray 9 C": (117, 120, 123),
    "Warm Red C": (245, 80, 56),
    "Rubine Red C": (206, 0, 88),
    "Rhodamine Red C": (225, 0, 152),
    "Purple C": (187, 41, 187),
    "Violet C": (68, 0, 153),
    "Blue 072 C": (16, 6, 159),
    "Reflex Blue C": (0, 20, 137),
    "Process Blue C": (0, 133, 202),
    "Green C": (0, 171, 132),
    "Orange 021 C": (254, 80, 0),
    "Yellow C": (254, 221, 0),
}


def hex_to_rgb(hex_str):
    """Convert hex color string to RGB tuple."""
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))


def recolor_product_fast(image, mask, target_rgb):
    """
    Vectorized version of recolor using NumPy for speed.
    """
    img_array = np.array(image).astype(np.float64) / 255.0
    mask_array = np.array(mask)

    # Product mask from alpha
    if mask_array.ndim == 2:
        product_mask = mask_array > 128
    else:
        product_mask = mask_array[:, :, 3] > 128

    # Target HSV
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

    t_r, t_g, t_b = target_rgb[0] / 255.0, target_rgb[1] / 255.0, target_rgb[2] / 255.0
    target_hsv = rgb_to_hsv(np.array([[[t_r, t_g, t_b]]]))[0, 0]

    # Convert image to HSV
    rgb_img = img_array[:, :, :3]
    hsv_img = rgb_to_hsv(rgb_img)

    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

    # Build exclusion masks for regions we want to preserve
    is_white = (s < 0.10) & (v > 0.85)
    is_black = v < 0.15
    is_gray = s < 0.08
    preserve = is_white | is_black | is_gray

    # Pixels to recolor: in product mask and not preserved
    recolor_mask = product_mask & ~preserve

    # Blend strength based on saturation
    blend = np.clip(s / 0.25, 0, 1)

    # Apply new hue and blended saturation
    hsv_result = hsv_img.copy()
    hsv_result[recolor_mask, 0] = target_hsv[0]
    hsv_result[recolor_mask, 1] = (
        target_hsv[1] * blend[recolor_mask] + s[recolor_mask] * (1 - blend[recolor_mask])
    )
    # Value (brightness) stays the same

    # Convert back to RGB
    rgb_result = hsv_to_rgb(hsv_result)
    result = (rgb_result * 255).astype(np.uint8)

    # Preserve alpha if present
    if img_array.shape[2] == 4:
        alpha = (img_array[:, :, 3] * 255).astype(np.uint8)
        result = np.dstack([result, alpha])

    return Image.fromarray(result)


def color_swap(image, pantone_choice, hex_color, model_name):
    """Process a single image: remove background to get mask, then recolor."""
    if image is None:
        return None, None

    # Determine target color
    target_rgb = None
    if pantone_choice and pantone_choice in PANTONE_COLORS and PANTONE_COLORS[pantone_choice]:
        target_rgb = PANTONE_COLORS[pantone_choice]
    elif hex_color:
        try:
            target_rgb = hex_to_rgb(hex_color)
        except (ValueError, IndexError):
            return None, None

    if target_rgb is None:
        return None, None

    # Open image
    img = Image.open(image).convert("RGBA")

    # Use rembg to get the mask (background removed version has alpha)
    session = get_session(model_name)
    removed = remove(img, session=session)

    # Use the alpha channel from removed as the product mask
    mask = np.array(removed)

    # Recolor the original image using the mask
    recolored = recolor_product_fast(img, mask, target_rgb)

    # Create a version with background removed + recolored
    recolored_array = np.array(recolored)
    removed_array = np.array(removed)
    # Apply the original alpha to the recolored image
    if recolored_array.shape[2] == 3:
        recolored_nobg = np.dstack([recolored_array, removed_array[:, :, 3]])
    else:
        recolored_nobg = recolored_array.copy()
        recolored_nobg[:, :, 3] = removed_array[:, :, 3]

    recolored_with_bg = Image.fromarray(recolored_array[:, :, :3] if recolored_array.shape[2] == 4 else recolored_array)
    recolored_without_bg = Image.fromarray(recolored_nobg.astype(np.uint8))

    return recolored_with_bg, recolored_without_bg

# Available rembg models
MODELS = [
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "silueta",
    "isnet-general-use",
    "isnet-anime",
    "birefnet-general",
    "birefnet-general-lite",
    "birefnet-portrait",
    "birefnet-dis",
    "birefnet-hrsod",
    "birefnet-cod",
    "birefnet-massive",
]

DEFAULT_MODEL = "isnet-general-use"


def remove_background_batch(images, model_name):
    if not images:
        return [], None

    session = get_session(model_name)
    results = []
    temp_dir = tempfile.mkdtemp()

    for i, img_path in enumerate(images):
        img = Image.open(img_path)
        result = remove(img, session=session)
        out_path = os.path.join(temp_dir, f"result_{i + 1}.png")
        result.save(out_path)
        results.append((out_path, os.path.basename(img_path)))

    # Create a zip of all results for download
    zip_path = os.path.join(temp_dir, "results.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for out_path, orig_name in results:
            name = os.path.splitext(orig_name)[0] + "_no_bg.png"
            zf.write(out_path, name)

    gallery_images = [(path, name) for path, name in results]
    return gallery_images, zip_path


with gr.Blocks(title="Product Image Tools") as demo:
    gr.Markdown("# Product Image Tools")

    with gr.Tabs():
        with gr.TabItem("Background Remover"):
            gr.Markdown("Upload one or more product images to remove their backgrounds.")

            with gr.Row():
                with gr.Column():
                    input_images = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        label="Upload Images",
                    )
                    model_dropdown = gr.Dropdown(
                        choices=MODELS,
                        value=DEFAULT_MODEL,
                        label="Model",
                    )
                    remove_btn = gr.Button("Remove Background", variant="primary")

                with gr.Column():
                    output_gallery = gr.Gallery(label="Results", columns=3)
                    download_btn = gr.File(label="Download All (ZIP)")

            remove_btn.click(
                fn=remove_background_batch,
                inputs=[input_images, model_dropdown],
                outputs=[output_gallery, download_btn],
            )

        with gr.TabItem("Color Swap"):
            gr.Markdown("Upload a product image and change its color to a Pantone swatch or custom color.")

            with gr.Row():
                with gr.Column():
                    swap_input = gr.File(
                        file_count="single",
                        file_types=["image"],
                        label="Upload Product Image",
                    )
                    pantone_dropdown = gr.Dropdown(
                        choices=list(PANTONE_COLORS.keys()),
                        value="",
                        label="Pantone Color (select one)",
                    )
                    hex_input = gr.Textbox(
                        label="Or enter a hex color (e.g. #FF5733)",
                        placeholder="#FF5733",
                    )
                    swap_model_dropdown = gr.Dropdown(
                        choices=MODELS,
                        value=DEFAULT_MODEL,
                        label="Model (for product detection)",
                    )
                    swap_btn = gr.Button("Swap Color", variant="primary")

                with gr.Column():
                    swap_output_bg = gr.Image(label="Recolored (with background)", type="pil")
                    swap_output_nobg = gr.Image(label="Recolored (background removed)", type="pil")

            swap_btn.click(
                fn=color_swap,
                inputs=[swap_input, pantone_dropdown, hex_input, swap_model_dropdown],
                outputs=[swap_output_bg, swap_output_nobg],
            )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
