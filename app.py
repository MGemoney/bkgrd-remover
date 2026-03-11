import os
import tempfile
import zipfile

import gradio as gr
from rembg import remove, new_session
from PIL import Image

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

    session = new_session(model_name)
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


with gr.Blocks(title="Product Background Remover") as demo:
    gr.Markdown("# Product Background Remover")
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
