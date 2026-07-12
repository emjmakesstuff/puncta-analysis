🔬 Puncta Analysis
Automated cell & puncta counting from fluorescence microscopy images.

Puncta Analysis detects and counts fluorescent puncta (or cells) in microscopy images automatically — no manual clicking required. It estimates detection settings from each image, lets you fine-tune interactively, and compares results against hand counts for validation.

✨ Features
Automatic counting — detects puncta using adaptive, image-specific settings
Interactive tuning — adjust detection with live sliders and instant preview
Dual measurements — puncta count and bright-area, each independently tunable
Compare to hand counts — overlay manual annotations to verify accuracy
Batch processing — analyze whole folders of images at once
Point-and-click app — a friendly web interface, no coding needed
👩‍🔬 For Users (no coding required)
One-time setup
Install Python. Download and install Anaconda (includes Python). Accept the defaults during installation.

Download Puncta Analysis. On the GitHub page, click the green Code button → Download ZIP. Unzip it somewhere memorable (e.g., your Documents folder).

Run the setup script (one time only):

Mac: Double-click setup_mac.command in the unzipped folder. (If macOS blocks it: right-click → Open → Open.)
This creates the environment and installs everything. It may take a few minutes. When it says it's done, you're ready.
Running the app
Mac: Double-click launch_mac.command.
Windows: Double-click launch_windows.bat.
Your web browser will open automatically with the Puncta Analysis app. (A terminal window will also open — leave it running while you use the app. Close it when you're done.)

Using the app
The app guides you through four steps (tabs at the top):

🏠 Home — an overview of the tool.

🗂️ Image Selection — click Choose folder… and select the folder containing your images. Organize your data so each image is in its own subfolder, along with any hand-count ROI files (.zip) for that image. Images without hand counts work fine too.

📊 Results — the tool automatically counts every image and shows a summary table plus preview thumbnails. Click Enlarge to inspect any image closely. Click Process All & Export to save the results.

🎛️ Manual Tuning (optional) — not happy with an automatic result? Adjust the detection for any image with live sliders:

Brightness threshold (sensitivity): how bright a spot must be to count
Puncta size (σ): the expected size of puncta
Area threshold: which pixels count as "bright" (shown as the red region)
Toggle your hand counts on/off to compare. Click Save to keep your settings, then return to Results.

💻 For Developers
Installation (editable)
bash
Copy code
git clone https://github.com/YOUR_USERNAME/puncta-analysis.git
cd puncta-analysis
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
Command-line interface
css
Copy code
# Batch-analyze a folder of images
puncta analyze --config config.yaml

# Batch-analyze, tuning each image interactively first
puncta analyze --config config.yaml --tune-each

# Interactive tuning of a single image
puncta tune --image path/to/image.tif

# Validate auto + tuned detection against manual counts
puncta validate --config config.yaml --spatial
Copy config.example.yaml to config.yaml and edit it to set input/output folders, channel, detection parameters, and output options.

The web app
text
Copy code
streamlit run app.py
How it works
Detection (count): a Difference-of-Gaussians (DoG) filter enhances puncta-sized blobs; local maxima above a sensitivity threshold are counted.
Area: a Gaussian Mixture Model (GMM) on pixel intensities finds a background/signal cutoff; pixels above it are the bright area.
Parameter estimation: puncta size is estimated from multi-scale blob detection; the sensitivity threshold from an Otsu cut on the filtered image — so settings adapt to each image without manual input.
Testing
text
Copy code
pytest
📄 License
[Add your license here — e.g., MIT]

📚 Citation
[If this supports a publication, add citation info here]