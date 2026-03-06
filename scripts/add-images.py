#!/usr/bin/env python3

import re
import subprocess
from pathlib import Path
from collections import defaultdict
from urllib.parse import quote

import numpy as np
import imagehash
from PIL import Image
from skimage.metrics import structural_similarity as ssim

SRC_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
KEEP_EXTS = {".avif", ".webp"}

QUALITY = 50

PHASH_THRESHOLD = 6
SSIM_THRESHOLD = 0.92


# ----------------------------
# git helpers
# ----------------------------

def run(cmd):
    subprocess.run(cmd, check=True)


def run_out(cmd):
    return subprocess.check_output(cmd, text=True).strip()


def git_tracked(path):
    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def commit_paths(paths, msg_add, msg_up):
    tracked = any(git_tracked(p) for p in paths)

    run(["git", "add"] + [str(p) for p in paths])

    staged = run_out(["git", "diff", "--cached", "--name-only"])

    if staged:
        msg = msg_up if tracked else msg_add
        run(["git", "commit", "-m", msg])
        print(msg)


# ----------------------------
# utilities
# ----------------------------

def slugify(name):
    name = name.lower()
    name = re.sub(r"[^\w]+", "-", name)
    return name.strip("-")


# ----------------------------
# image conversion
# ----------------------------

def convert_to_avif(src):
    dst = Path(str(src) + ".avif")

    if dst.exists():
        return dst

    with Image.open(src) as im:
        im.save(dst, "AVIF", quality=QUALITY)

    return dst


# ----------------------------
# similarity detection
# ----------------------------

def compute_image_data(path):
    with Image.open(path) as im:
        ph = imagehash.phash(im)
        w, h = im.size
        pixels = w * h

        gray = im.convert("L").resize((256, 256))
        arr = np.array(gray)

    return {
        "path": path,
        "phash": ph,
        "pixels": pixels,
        "array": arr,
    }


def images_similar(a, b):
    if a["phash"] - b["phash"] > PHASH_THRESHOLD:
        return False

    score = ssim(a["array"], b["array"])
    return score >= SSIM_THRESHOLD


def deduplicate_images(images):

    data = [compute_image_data(i) for i in images]

    groups = []

    for img in data:

        placed = False

        for g in groups:
            if images_similar(img, g[0]):
                g.append(img)
                placed = True
                break

        if not placed:
            groups.append([img])

    chosen = []

    for g in groups:
        best = max(g, key=lambda x: x["pixels"])
        chosen.append(best["path"])

    return sorted(chosen)


# ----------------------------
# image scanning
# ----------------------------

def find_images(root):

    dirs = defaultdict(list)

    for p in root.rglob("*"):

        if not p.is_file():
            continue

        ext = p.suffix.lower()

        if ext in KEEP_EXTS:
            dirs[p.parent].append(p)

        elif ext in SRC_EXTS:
            avif = convert_to_avif(p)
            dirs[avif.parent].append(avif)

    for d in dirs:
        dirs[d] = sorted(set(dirs[d]))

    return dirs


# ----------------------------
# directory index generation
# ----------------------------

def write_dir_markdown(dir_path, images):

    lines = [f"# {dir_path.as_posix()}", ""]

    for img in images:

        name = img.name
        label = escape_md_text(name)
        url = encode_md_url(name)

        lines += [
            f"## {label}",
            "",
            f"![{label}]({url})",
            "",
        ]

    (dir_path / "readme.md").write_text("\n".join(lines))


def write_dir_html(dir_path, images):

    lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<body>",
        f"<h1>{dir_path.as_posix()}</h1>",
    ]

    for img in images:

        name = img.name
        slug = slugify(name)

        lines += [
            f'<section id="{slug}">',
            f'<h2><a href="#{slug}">{name}</a></h2>',
            f'<img src="{name}" loading="lazy">',
            "</section>",
        ]

    lines += [
        "</body>",
        "</html>",
    ]

    (dir_path / "index.html").write_text("\n".join(lines))


# ----------------------------
# root index generation
# ----------------------------

def build_tree(dirs):

    tree = {}

    for d in dirs:

        node = tree

        for part in d.parts:
            node = node.setdefault(part, {})

    return tree


def escape_md_text(text: str) -> str:
    # escape characters that break markdown link labels
    return text.replace("]", r"\]")


def encode_md_url(url: str) -> str:
    # encode spaces, ), etc.
    return quote(url, safe="/")


def render_tree_md(node, path="", image_dirs=set()):
    lines = []

    for name, child in sorted(node.items()):

        new_path = f"{path}/{name}" if path else name
        has_images = Path(new_path) in image_dirs

        label = escape_md_text(name)
        link = encode_md_url(f"{new_path}/readme.md")

        if child:

            if has_images:
                lines.append(f"- [{label}]({link})")
            else:
                lines.append(f"- {label}")

            sub = render_tree_md(child, new_path, image_dirs)
            lines += ["  " + s for s in sub]

        else:

            lines.append(f"- [{label}]({link})")

    return lines


def render_tree_html(node, path="", image_dirs=set()):

    lines = ["<ul>"]

    for name, child in sorted(node.items()):

        new_path = f"{path}/{name}" if path else name
        has_images = Path(new_path) in image_dirs

        if child:

            if has_images:
                lines.append(f'<li><a href="{new_path}/index.html">{name}</a>')
            else:
                lines.append(f"<li>{name}")

            lines += render_tree_html(child, new_path, image_dirs)
            lines.append("</li>")

        else:

            lines.append(f'<li><a href="{new_path}/index.html">{name}</a></li>')

    lines.append("</ul>")
    return lines


def write_root_indexes(dirs):

    tree = build_tree(dirs)
    image_dirs = set(dirs.keys())

    md = ["# alchi-faces", ""]
    md += render_tree_md(tree, "", image_dirs)

    Path("readme.md").write_text("\n".join(md))

    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<body>",
        "<h1>alchi-faces</h1>",
    ]

    html += render_tree_html(tree, "", image_dirs)

    html += [
        "</body>",
        "</html>",
    ]

    Path("index.html").write_text("\n".join(html))


# ----------------------------
# main
# ----------------------------

def main():

    root = Path(".")

    dirs = find_images(root)

    for d in sorted(dirs):

        images = deduplicate_images(dirs[d])

        write_dir_markdown(d, images)
        write_dir_html(d, images)

        files = images + [d / "readme.md", d / "index.html"]

        rel = d.as_posix()

        commit_paths(
            files,
            f"add {rel}",
            f"up {rel}",
        )

    write_root_indexes(dirs)

    commit_paths(
        [Path("readme.md"), Path("index.html")],
        "add index.html readme.md",
        "up index.html readme.md",
    )


if __name__ == "__main__":
    main()
