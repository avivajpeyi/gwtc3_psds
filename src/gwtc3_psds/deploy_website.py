import glob
import argparse
import tempfile
import shutil
import os
import webbrowser
from datetime import datetime

IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "event_data")


def build_psd_gallery(input_dir, output_dir, cols=3, image_size="medium"):
    """Generate a static HTML gallery of PSDs from input_dir into output_dir."""
    psd_files = sorted(glob.glob(os.path.join(input_dir, "*_psd.png")))

    if not psd_files:
        print(f"⚠️  No *_psd.png files found in {input_dir}")
        return

    # Image size options
    size_configs = {
        "small": {"max_width": "200px", "thumb_width": "180px"},
        "medium": {"max_width": "300px", "thumb_width": "280px"},
        "large": {"max_width": "400px", "thumb_width": "380px"},
        "xlarge": {"max_width": "500px", "thumb_width": "480px"}
    }

    config = size_configs.get(image_size, size_configs["medium"])

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>LVK PSD Gallery</title>
      <style>
        * {{
          box-sizing: border-box;
        }}
        body {{
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          margin: 0;
          padding: 20px;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          min-height: 100vh;
        }}
        .container {{
          max-width: 1200px;
          margin: 0 auto;
          background: white;
          border-radius: 12px;
          box-shadow: 0 10px 30px rgba(0,0,0,0.2);
          overflow: hidden;
        }}
        .header {{
          background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
          color: white;
          padding: 30px;
          text-align: center;
        }}
        h1 {{
          margin: 0;
          font-size: 2.5em;
          font-weight: 300;
        }}
        .subtitle {{
          margin-top: 10px;
          opacity: 0.9;
          font-size: 1.1em;
        }}
        .gallery-container {{
          padding: 30px;
        }}
        .gallery {{
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
          gap: 25px;
          margin-top: 20px;
        }}
        .psd-item {{
          background: #f8f9fa;
          border-radius: 12px;
          padding: 20px;
          text-align: center;
          transition: all 0.3s ease;
          border: 2px solid transparent;
          position: relative;
          overflow: hidden;
        }}
        .psd-item::before {{
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          height: 4px;
          background: linear-gradient(90deg, #667eea, #764ba2);
          transform: scaleX(0);
          transition: transform 0.3s ease;
        }}
        .psd-item:hover {{
          transform: translateY(-5px);
          box-shadow: 0 15px 35px rgba(0,0,0,0.15);
          border-color: #667eea;
        }}
        .psd-item:hover::before {{
          transform: scaleX(1);
        }}
        .image-container {{
          position: relative;
          margin-bottom: 15px;
        }}
        img {{
          max-width: {config["max_width"]};
          width: 100%;
          height: auto;
          border-radius: 8px;
          box-shadow: 0 4px 12px rgba(0,0,0,0.1);
          transition: transform 0.3s ease;
          cursor: pointer;
        }}
        img:hover {{
          transform: scale(1.02);
        }}
        .caption {{
          font-size: 16px;
          font-weight: 500;
          color: #2c3e50;
          margin-top: 10px;
          word-break: break-word;
        }}
        .stats {{
          background: #e9ecef;
          padding: 20px;
          text-align: center;
          color: #495057;
          font-size: 14px;
        }}
        .lightbox {{
          display: none;
          position: fixed;
          z-index: 1000;
          left: 0;
          top: 0;
          width: 100%;
          height: 100%;
          background-color: rgba(0,0,0,0.9);
          cursor: pointer;
        }}
        .lightbox-content {{
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          max-width: 90%;
          max-height: 90%;
          border-radius: 8px;
        }}
        .close {{
          position: absolute;
          top: 20px;
          right: 35px;
          color: white;
          font-size: 40px;
          font-weight: bold;
          cursor: pointer;
        }}
        @media (max-width: 768px) {{
          .gallery {{
            grid-template-columns: 1fr;
            gap: 20px;
          }}
          .container {{
            margin: 10px;
            border-radius: 8px;
          }}
          .header {{
            padding: 20px;
          }}
          h1 {{
            font-size: 2em;
          }}
        }}
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>LVK Power Spectral Densities</h1>
          <div class="subtitle">Gravitational Wave Event Analysis</div>
          <div style="margin-top: 10px; opacity: 0.8; font-size: 0.9em;">
            Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
          </div>
        </div>

        <div class="gallery-container">
          <div class="gallery">
    """

    # Add images to gallery
    for psd in psd_files:
        gw_name = os.path.basename(psd).replace("_psd.png", "")
        # Clean up the name for better display
        display_name = gw_name.replace("_", " ").title()

        html_content += f"""
            <div class="psd-item">
              <div class="image-container">
                <img src="{os.path.basename(psd)}" alt="{display_name}" onclick="openLightbox('{os.path.basename(psd)}', '{display_name}')">
              </div>
              <div class="caption">{display_name}</div>
            </div>
        """

    html_content += f"""
          </div>
        </div>

        <div class="stats">
          <strong>{len(psd_files)} PSD plots</strong> • 
          Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M")} • 
          Click images to view full size
        </div>
      </div>

      <!-- Lightbox -->
      <div id="lightbox" class="lightbox" onclick="closeLightbox()">
        <span class="close">&times;</span>
        <img class="lightbox-content" id="lightbox-img">
      </div>

      <script>
        function openLightbox(src, alt) {{
          document.getElementById('lightbox').style.display = 'block';
          document.getElementById('lightbox-img').src = src;
          document.getElementById('lightbox-img').alt = alt;
          document.body.style.overflow = 'hidden';
        }}

        function closeLightbox() {{
          document.getElementById('lightbox').style.display = 'none';
          document.body.style.overflow = 'auto';
        }}

        // Close lightbox with Escape key
        document.addEventListener('keydown', function(e) {{
          if (e.key === 'Escape') {{
            closeLightbox();
          }}
        }});
      </script>
    </body>
    </html>
    """

    os.makedirs(output_dir, exist_ok=True)

    # Copy images into output dir
    for psd in psd_files:
        shutil.copy(psd, os.path.join(output_dir, os.path.basename(psd)))

    # Write HTML file
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html_content)

    print(f"✅ Gallery built with {len(psd_files)} PSDs at {output_dir}/index.html")


def deploy_with_ghp_import(build_dir):
    """Deploy the build dir to GitHub Pages using ghp-import."""
    try:
        result = os.system(f"ghp-import -n -p -f {build_dir}")
        if result == 0:
            print("🚀 Successfully deployed to GitHub Pages")
        else:
            print("❌ Deployment failed - check if ghp-import is installed")
    except Exception as e:
        print(f"❌ Deployment error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Build and optionally deploy PSD gallery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py                              # Build gallery with default settings
  python script.py --show                       # Build and open locally
  python script.py --size large --cols 2        # Larger images, 2 columns
  python script.py --upload --outdir ./gallery  # Build to specific dir and deploy
        """
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=IMAGE_DIR,
        help="Directory containing *_psd.png files (default: event_data/)"
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: temporary directory)"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Deploy to GitHub Pages using ghp-import"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the gallery in browser and keep output directory"
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Number of columns in gallery (default: 1)"
    )
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large", "xlarge"],
        default="large",
        help="Image size: small (200px), medium (300px), large (400px), xlarge (500px)"
    )

    args = parser.parse_args()

    # print parsed arguments for debugging
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {args.outdir}")
    print(f"Upload: {args.upload}")
    print(f"Show: {args.show}")
    print(f"Columns: {args.cols}")
    print(f"Image Size: {args.size}")


    # Handle output directory
    if args.outdir is None:
        tmpdir = tempfile.mkdtemp(prefix="psd_gallery_")
        keep_outdir = args.show  # Keep if showing locally
    else:
        tmpdir = os.path.abspath(args.outdir)
        keep_outdir = True

    try:
        build_psd_gallery(args.input_dir, tmpdir, args.cols, args.size)

        if args.upload:
            deploy_with_ghp_import(tmpdir)

        if args.show:
            index_path = os.path.join(tmpdir, "index.html")
            if os.path.exists(index_path):
                webbrowser.open(f"file://{index_path}")
                print(f"📂 Gallery opened from: {tmpdir}")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if not keep_outdir:
            shutil.rmtree(tmpdir)
            print("🧹 Cleaned up temporary files")


if __name__ == "__main__":
    main()