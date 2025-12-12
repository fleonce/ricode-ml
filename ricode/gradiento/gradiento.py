import functools
import json
import os
import subprocess
from collections import defaultdict

import gradio as gr
import more_itertools

from ricode.gradiento.args import GradientoArgs


def func(a: str):
    return a[::-1]


title_css = """
h1 .centered {
    text-align: center;
    display: block;
}

h2 .centered {
    text-align: center;
    display: block;
}
"""

args = GradientoArgs(os.environ["GRADIENTO_PATH"])

model_folders = []
folder_parents = defaultdict(list)
roots = []
for root, folders, files in os.walk(args.path):
    if (
        "experiment.json" in files
        and root not in model_folders
        and ("loss.pdf" in files or "loss.svg" in files or len(folders) > 0)
    ):
        for known_root in roots:
            if root.startswith(known_root):
                folder_parents[root].append(known_root)
        model_folders.append(root)
    if len(files) == 0 and len(folders) > 0:
        if os.path.dirname(root) in roots:
            roots.remove(os.path.dirname(root))
        roots.append(root)

    for subfolder in list(folders):
        if subfolder.startswith("step-"):
            folders.remove(subfolder)
roots = roots[1:]
folder_parents = {k: tuple(sorted(v, key=len)) for k, v in folder_parents.items()}
parents_to_folders = {v: [] for v in folder_parents.values()}
for key, value in folder_parents.items():
    parents_to_folders[value].append(key)
parents_to_folders = {k: v for k, v in parents_to_folders.items() if len(v) > 0}
parents_to_folders = {k: list(sorted(v)) for k, v in parents_to_folders.items()}

folders_per_row = 5
key_ordering = list(sorted(parents_to_folders.keys(), key=len))


def _load_image(button, folder: str):
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
        json_config = json.load(f)["training"]

    return [
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=True, value=svg_path),
        gr.update(visible=False, value=json_config),
    ]


def _hide_image():
    return [
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    ]


with gr.Blocks() as demo:
    gr.Markdown("# Gradiento", elem_classes="centered")

    for key in key_ordering:
        gr.Markdown("## " + key[-1])

        for group_of_model_folders in more_itertools.batched(
            parents_to_folders[key], folders_per_row, strict=False
        ):
            with gr.Row(equal_height=False):
                data = []

                for model_folder in group_of_model_folders:
                    with gr.Column(scale=1):
                        model_name = model_folder.replace(key[-1], "")
                        if model_name.startswith("/"):
                            model_name = model_name[1:]
                        model_button = gr.Button(model_name, variant="huggingface")
                        hide_model_button = gr.Button(
                            model_name, variant="primary", visible=False
                        )
                        data.append(
                            (
                                model_folder,
                                model_button,
                                hide_model_button,
                            )
                        )
            with gr.Row():
                for model_folder, model_button, hide_model_button in data:
                    model_image = gr.Image(visible=False)
                    model_config = gr.JSON(visible=False, open=True)

                    model_button.click(
                        functools.partial(_load_image, folder=model_folder),
                        inputs=[model_button],
                        outputs=[
                            model_button,
                            hide_model_button,
                            model_image,
                            model_config,
                        ],
                    )
                    hide_model_button.click(
                        _hide_image,
                        inputs=[],
                        outputs=[
                            model_button,
                            hide_model_button,
                            model_image,
                            model_config,
                        ],
                    )

demo.launch(enable_monitoring=True, css=title_css, footer_links=[])
