from base64 import b64encode
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torchvision import transforms


CHECKPOINT_PATH = Path(__file__).resolve().parent / "models" / "best_cnn_augmented.pth"
CATEGORIES = {
    "buildings": "Buildings, architecture, and city skylines.",
    "forest": "Woodland scenes with trees and vegetation.",
    "glacier": "Glaciers and landscapes dominated by ice and snow.",
    "mountain": "Mountain peaks, slopes, and rocky landscapes.",
    "sea": "Seas, coastlines, and open water.",
    "street": "Urban streets, roads, and surrounding buildings.",
}
TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), # 3 признака (RGB) - в 16 обучаемых фильтров
            nn.ReLU(),
            nn.MaxPool2d(2),                            # уменьшаем h и w в 2 раза
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()                     # признаки - в вектор
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),               # cжимаем вектор из 16384 признаков до 128
            nn.ReLU(),
            nn.Linear(128, 6),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        return self.classifier(x)


@st.cache_resource
def load_model():
    # CPU подходит и для локального запуска, и для Community Cloud.
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    classes = list(checkpoint["classes"])
    if classes != list(CATEGORIES):
        raise ValueError("The checkpoint contains an unexpected class order.")
    model = SimpleCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes



def show_prediction(uploaded):
    if uploaded.size > 10 * 1024 * 1024:
        st.error("Please upload an image smaller than 10 MB.")
        return

    try:
        with Image.open(uploaded) as source:
            if source.format not in ("JPEG", "PNG"):
                st.error("Please upload a valid JPG, JPEG, or PNG image.")
                return
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        st.error("This image could not be read. Please try another JPG, JPEG, or PNG file.")
        return

    try:
        model, classes = load_model()
    except (OSError, RuntimeError, KeyError, ValueError):
        st.error("The model could not be loaded. Please contact the app owner.")
        return

    # Softmax нужен для отображения вероятностей, модель возвращает логиты.
    with torch.no_grad():
        probabilities = model(TRANSFORM(image).unsqueeze(0)).softmax(dim=1)[0].tolist()
    best_index = max(range(len(classes)), key=lambda index: probabilities[index])
    preview, result = st.columns([1, 1.15], gap="large")
    with preview:
        st.image(image, caption="Uploaded photo", width="stretch")
    with result:
        st.subheader("Prediction")
        category, confidence = st.columns(2)
        category.metric("Category", classes[best_index].title())
        confidence.metric("Confidence", f"{probabilities[best_index]:.1%}")
        st.markdown("**Class probabilities**")
        rows = [
            {"Category": name.title(), "Probability": probability * 100}
            for name, probability in zip(classes, probabilities)
        ]
        rows.sort(key=lambda row: row["Probability"], reverse=True)
        st.bar_chart(
            rows, x="Category", y="Probability", horizontal=True,
            sort="-Probability", color="#B5A6D3", height=310,
            x_label="Category", y_label="Probability (%)",
        )


def main():
    st.set_page_config(page_title="Scene Classification", layout="wide")
    background_path = Path(__file__).resolve().parent / "assets" / "cat-background.jpg"
    background_image = "none"
    if background_path.is_file():
        # Встроенное изображение работает и локально, и в Community Cloud.
        encoded_image = b64encode(background_path.read_bytes()).decode("ascii")
        background_image = f'url("data:image/jpeg;base64,{encoded_image}")'

    st.html("""
        <style>
        .stApp {
            background: #0D0E12;
            --scene-background: __BACKGROUND_IMAGE__;
        }
        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            z-index: 0;
            pointer-events: none;
            background-image:
                linear-gradient(90deg, rgba(13,14,18,.97) 0%,
                    rgba(13,14,18,.86) 35%, rgba(13,14,18,.35) 100%),
                linear-gradient(180deg, rgba(13,14,18,.12), rgba(13,14,18,.72)),
                var(--scene-background);
            background-size: cover;
            background-position: right center;
            background-repeat: no-repeat;
        }
        [data-testid="stAppViewContainer"] {
            z-index: 1;
            background: transparent;
        }
        [data-testid="stHeader"] {background: rgba(13,14,18,.85);}
        [data-testid="stMainBlockContainer"] {
            max-width: 1240px;
            padding: 3rem 2rem 2rem;
        }
        h1 {letter-spacing: -.04em;}
        [data-testid="stCaptionContainer"] {color: #C2BFCB;}
        [class*="st-key-category-"], .st-key-upload-panel {
            background: rgba(27,28,36,.82);
            border: 1px solid rgba(181,166,211,.20);
            border-radius: 18px;
            box-shadow: 0 8px 24px rgba(0,0,0,.12);
        }
        [data-testid="stMetric"] {
            background: rgba(27,28,36,.76);
            border: 1px solid rgba(181,166,211,.16);
            border-radius: 16px;
            padding: 1rem 1.2rem;
        }
        [data-testid="stMetricValue"] {color: #DDD5EB;}
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(40,39,51,.88);
            border: 1px dashed rgba(181,166,211,.55);
            border-radius: 14px;
        }
        [data-testid="stImage"] img {border-radius: 12px;}
        @media (max-width: 768px) {
            .stApp::before {
                background-image:
                    linear-gradient(rgba(13,14,18,.84), rgba(13,14,18,.94)),
                    var(--scene-background);
            }
            [data-testid="stMainBlockContainer"] {padding: 2rem 1rem;}
            [class*="st-key-category-"], .st-key-upload-panel {
                background: rgba(27,28,36,.94);
            }
        }
        </style>
    """.replace("__BACKGROUND_IMAGE__", background_image))
    st.title("Scene Classification")
    st.write("A custom CNN for six kinds of landscapes and urban scenes.")
    st.caption("Upload a photo and explore how the model sees it.")

    accuracy, sample_count, model_info = st.columns(3)
    accuracy.metric("Test accuracy", "82.7%")
    sample_count.metric("Test images", "3,000")
    model_info.metric("Model", "SimpleCNN")
    st.caption("Built with PyTorch · evaluated on an independent test set")

    st.subheader("Supported categories")
    names = list(CATEGORIES)
    for start in (0, 3):
        columns = st.columns(3)
        for column, name in zip(columns, names[start:start + 3]):
            with column.container(border=True, key=f"category-{name}"):
                st.markdown(f"**{name.title()}**")
                st.caption(CATEGORIES[name])

    st.write("")
    with st.container(border=True, key="upload-panel"):
        st.subheader("Try a photo")
        st.caption("How to use: upload a JPG, JPEG, or PNG, then compare the prediction and class probabilities.")
        uploaded = st.file_uploader(
            "Choose your image", type=["jpg", "jpeg", "png"],
            help="JPG, JPEG, or PNG, up to 10 MB.",
        )
        if uploaded is not None:
            show_prediction(uploaded)

    st.divider()
    st.caption(
        "Limitations: This model knows only these six categories. Images of other objects "
        "may be classified incorrectly. Confidence does not guarantee correctness."
    )


if __name__ == "__main__":
    main()
