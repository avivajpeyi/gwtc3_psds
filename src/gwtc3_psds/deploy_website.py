import glob
import argparse
import tempfile
import shutil
import os
import webbrowser


IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "event_data")


def build_psd_gallery(input_dir, output_dir, cols=3):
    """Generate a static HTML gallery of PSDs from input_dir into output_dir."""
    psd_files = sorted(glob.glob(os.path.join(input_dir, "*_psd.png")))

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>LVK PSD Gallery</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          margin: 20px;
          background: #f9fafc;
        }
        h1 {
          text-align: center;
          margin-bottom: 20px;
        }
        table {
          width: 100%;
          border-collapse: collapse;
        }
        td {
          text-align: center;
          padding: 15px;
          vertical-align: top;
        }
        img {
          max-width: 250px;
          border-radius: 8px;
          box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }
        .caption {
          font-size: 14px;
          margin-top: 8px;
          color: #333;
        }
      </style>
    </head>
    <body>
      <h1>LVK Power Spectral Densities (PSDs)</h1>
      <table>
    """

    # arrange images in a table
    for i, psd in enumerate(psd_files):
        gw_name = os.path.basename(psd).replace("_psd.png", "")
        if i % cols == 0:
            html_content += "<tr>\n"
        html_content += f"""
          <td>
            <a href="{os.path.basename(psd)}" target="_blank">
              <img src="{os.path.basename(psd)}" alt="{gw_name}">
            </a>
            <div class="caption">{gw_name}</div>
          </td>
        """
        if i % cols == cols - 1:
            html_content += "</tr>\n"

    # close last row if incomplete
    if len(psd_files) % cols != 0:
        html_content += "</tr>\n"

    html_content += """
      </table>
    </body>
    </html>
    """

    os.makedirs(output_dir, exist_ok=True)

    # copy images into output dir
    for psd in psd_files:
        shutil.copy(psd, os.path.join(output_dir, os.path.basename(psd)))

    # write HTML file
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html_content)

    print(f"✅ Gallery built at {output_dir}/index.html")


def deploy_with_ghp_import(build_dir):
    """Deploy the build dir to GitHub Pages using ghp-import."""
    os.system(f"ghp-import -n -p -f {build_dir}")
    print("🚀 Deployed to GitHub Pages")




def main():
    parser = argparse.ArgumentParser(description="Build and optionally deploy PSD gallery")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=IMAGE_DIR,
        help="Directory containing input images (default: IMAGE_DIR)"
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: temporary directory)"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="If set, upload with ghp-import"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="If set, open the gallery locally and keep output directory"
    )
    args = parser.parse_args()

    # handle output directory
    if args.outdir is None:
        tmpdir = tempfile.mkdtemp(prefix="psd_gallery_")
        keep_outdir = False
    else:
        tmpdir = os.path.abspath(args.outdir)
        os.makedirs(tmpdir, exist_ok=True)
        keep_outdir = True

    try:
        build_psd_gallery(args.input_dir, tmpdir)

        if args.upload:
            deploy_with_ghp_import(tmpdir)

        if args.show:
            index_path = os.path.join(tmpdir, "index.html")
            if os.path.exists(index_path):
                webbrowser.open(f"file://{index_path}")
            keep_outdir = True

    finally:
        if not keep_outdir:
            shutil.rmtree(tmpdir)
            print("🧹 Cleaned up temporary files")

