# NST-AdaIN — Neural Style Transfer with Adaptive Instance Normalization

A PyTorch implementation of real-time arbitrary neural style transfer using **Adaptive Instance Normalization (AdaIN)**, based on the paper [*Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*](https://arxiv.org/abs/1703.06868) by Huang & Belongie (2017). The project includes both a training pipeline and a **Flask web application** for interactive style transfer.

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue?style=for-the-badge)](https://huggingface.co/spaces/sriKritarth/trial-2)

---

## 🎨 Live Demo

Try it out instantly — no setup required:

**👉 [https://huggingface.co/spaces/sriKritarth/trial-2](https://huggingface.co/spaces/sriKritarth/trial-2)**

Upload a content image and a style image, adjust the alpha strength, and get your stylized artwork in seconds.

---

## ✨ Features

- **Real-time arbitrary style transfer** — apply any style image to any content image in a single forward pass
- **Adjustable stylization strength** via the `alpha` parameter (0.0 = original content, 1.0 = full style)
- **Pre-trained VGG encoder** (`vgg_normalised.pth`) for feature extraction
- **Trainable decoder** that learns to invert AdaIN-transformed features back to image space
- **Flask web UI** with Bootstrap — upload content & style images and get results instantly
- **Heroku-ready** deployment via `Procfile` and `gunicorn`
- **Jupyter notebook** (`code.ipynb`) for experimentation

---

## 🧠 How It Works

AdaIN aligns the channel-wise mean and standard deviation of the **content features** to match those of the **style features**, without any style-specific parameters:

```
AdaIN(x, y) = σ(y) * ((x − μ(x)) / σ(x)) + μ(y)
```

The pipeline is:

```
Content Image ──► VGG Encoder ──► AdaIN ──► Decoder ──► Stylized Image
Style Image ───► VGG Encoder ──►
```

The encoder is a fixed, pre-trained VGG-19 (up to `relu4_1`). Only the **decoder** is trained, using a combined content and style loss computed over multiple VGG feature layers.

---

## 📁 Project Structure

```
NST-AdaIN/
├── app.py                  # Flask web application
├── train.py                # Training script
├── code.ipynb              # Jupyter notebook for exploration
├── requirements.txt        # Python dependencies
├── Procfile                # Heroku deployment config
├── vgg_normalised.pth      # Pre-trained normalised VGG encoder weights
├── utils/
│   ├── model.py            # VggEncoder and Decoder architectures
│   └── utils.py            # AdaIN, loss functions, dataset & transforms
├── templates/
│   └── index.html          # Flask HTML template
├── static/uploads/         # Uploaded and generated images
└── experiment/             # Training checkpoints and outputs
```

---

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/sriKritarth/NST-AdaIN.git
cd NST-AdaIN
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Web App

Make sure the trained decoder weights exist at `experiment/final_exp/decoder_final.pth`, then:

```bash
python app.py
```

Open your browser and visit `http://localhost:5000`.

**Using the interface:**
1. Upload a **content image** (the photo you want to stylize)
2. Upload a **style image** (the artwork whose style you want to apply)
3. Set an **alpha** value between `0.0` (pure content) and `1.0` (full style transfer)
4. Click **Transfer Style** and download your result

Supported image formats: `.png`, `.jpg`, `.jpeg`

---

## 🏋️ Training

To train the decoder from scratch, you need a content dataset (e.g., COCO) and a style dataset (e.g., WikiArt).

```bash
python train.py \
  --content_dataset path/to/content/images \
  --style_dataset path/to/style/images \
  --vgg vgg_normalised.pth \
  --experiment my_experiment \
  --epoch 20 \
  --batch_size 4 \
  --lr 1e-4 \
  --c_wt 1.0 \
  --s_wt 5.0
```

**Key training arguments:**

| Argument | Default | Description |
|---|---|---|
| `--content_dataset` | `test2017` | Path to content image folder |
| `--style_dataset` | `train_3` | Path to style image folder |
| `--vgg` | `vgg_normalised.pth` | Path to pre-trained VGG weights |
| `--experiment` | `experiment1` | Name of output experiment folder |
| `--epoch` | `1` | Number of training epochs |
| `--batch_size` | `4` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--lr_decay` | `5e-5` | Learning rate decay factor |
| `--c_wt` | `1.0` | Content loss weight |
| `--s_wt` | `5.0` | Style loss weight |
| `--final_size` | `256` | Output image size |
| `--save_interval` | `5` | Save checkpoint every N epochs |
| `--resume` | `False` | Resume from checkpoint |
| `--decoder_path` | `None` | Path to decoder checkpoint (for resume) |
| `--optimizer_path` | `None` | Path to optimizer checkpoint (for resume) |

Checkpoints are saved under `experiment/<experiment_name>/` and sample outputs are saved every `--save_interval` epochs.

---

## 🌐 Deployment

### Hugging Face Spaces (Live)

The app is live on Hugging Face Spaces:

**👉 [https://huggingface.co/spaces/sriKritarth/trial-2](https://huggingface.co/spaces/sriKritarth/trial-2)**

### Heroku

The repository includes a `Procfile` for Heroku deployment using `gunicorn`:

```
web: gunicorn app:app
```

Deploy with:

```bash
heroku create
git push heroku main
```

---

## 📦 Dependencies

| Package | Version |
|---|---|
| torch | 2.11.0 |
| torchvision | 0.26.0 |
| Flask | 3.1.3 |
| Flask-Bootstrap | 3.3.7.1 |
| Flask-WTF | 1.3.0 |
| Werkzeug | 3.1.4 |
| WTForms | 3.2.1 |
| Pillow | 12.0.0 |
| numpy | 1.26.4 |
| tqdm | 4.67.3 |
| gunicorn | 26.0.0 |

---

## 📄 Reference

> Huang, X., & Belongie, S. (2017). **Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization**. *ICCV 2017*. [arXiv:1703.06868](https://arxiv.org/abs/1703.06868)

---

## 🙏 Acknowledgements

- Original AdaIN paper by Xun Huang and Serge Belongie
- Pre-trained VGG weights from the original authors
- COCO dataset for content images
- WikiArt dataset for style images
