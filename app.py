import os

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

DEFAULT_MODEL = "u2net"


def remove_background(image, model_name):
    if image is None:
        return None

    session = new_session(model_name)
    result = remove(image, session=session)
    return result


with gr.Blocks(title="Product Background Remover") as demo:
    gr.Markdown("# Product Background Remover")
    gr.Markdown("Upload a product image to remove its background.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            model_dropdown = gr.Dropdown(
                choices=MODELS,
                value=DEFAULT_MODEL,
                label="Model",
            )
            remove_btn = gr.Button("Remove Background", variant="primary")

        with gr.Column():
            output_image = gr.Image(type="pil", label="Result")

    remove_btn.click(
        fn=remove_background,
        inputs=[input_image, model_dropdown],
        outputs=output_image,
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
