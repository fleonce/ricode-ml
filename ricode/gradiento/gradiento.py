import os
import subprocess

import gradio as gr
import more_itertools

from ricode.gradiento.args import GradientoArgs


def func(a: str):
    return a[::-1]


title_css = """
h1 {
    text-align: center;
    display: block;
}

h2 {
    text-align: center;
    display: block;
}
"""

# inkscape \
#   --without-gui \
#   --file=input.pdf \
#   --export-plain-svg=output.svg
args = GradientoArgs(os.environ["GRADIENTO_PATH"])


model_folders = []
for root, folders, files in os.walk(args.path):
    if "loss.pdf" in files and root not in model_folders:
        model_folders.append(root)
    for subfolder in list(folders):
        if subfolder.startswith("step-"):
            folders.remove(subfolder)
folders_per_row = 4


def _load_image(folder: str):
    svg_path = os.path.join(folder, "loss.svg")
    if not os.path.exists(svg_path):
        # model is still training, find the latest subdirectory
        stepcount = 0
        for subdir in os.listdir(folder):
            if not os.path.isdir(os.path.join(folder, subdir)):
                continue
            if not subdir.startswith("step-") or not os.path.exists(
                os.path.join(folder, subdir, "loss.svg")
            ):
                continue
            subdir_stepcount = int(subdir[len("step-") :])
            if subdir_stepcount > stepcount:
                stepcount = subdir_stepcount
        if stepcount == 0:
            pdf_path = os.path.join(folder, "loss.pdf")
            if os.path.exists(pdf_path):
                proc = subprocess.run(
                    [
                        "inkscape",
                        "--without-gui",
                        "--file=" + pdf_path,
                        "--export-plain-svg=" + svg_path,
                    ]
                )
                if proc.returncode != 0:
                    raise ValueError(proc.stderr)
        else:
            svg_path = os.path.join(folder, "step-" + str(stepcount), "loss.svg")

    with open(os.path.join(folder, "experiment.json")) as f:
        json_config = f.read()

    return [
        gr.update("img", visible=True, value=svg_path),
        gr.update("config", visible=True, value=json_config),
    ]
    return os.path.join(folder, "loss.svg")


# @with_dataclass(GradientoArgs)
# def main(args: GradientoArgs):
# if __name__ == "__main__":
with gr.Blocks() as demo:
    gr.Markdown("# Gradiento")

    model_buttons: list[tuple[gr.Button, str]] = []
    with gr.Row() as row:
        for group_of_model_folders in more_itertools.distribute(
            folders_per_row, model_folders
        ):
            with gr.Column(scale=1):
                for model_folder in group_of_model_folders:
                    model_button = gr.Button(model_folder, variant="huggingface")
                    model_buttons.append((model_button, model_folder))

    gr.Markdown("## Training Details")
    with gr.Row(equal_height=False) as row:
        image = gr.Image(visible=False, elem_id="img")
        config = gr.JSON(visible=False, elem_id="config", height=None, open=True)
    for model_button, model_folder in model_buttons:
        model_button.click(
            _load_image,
            inputs=[model_button],
            outputs=[image, config],
        )

demo.launch(enable_monitoring=True, css=title_css, footer_links=[])
