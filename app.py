import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from src.model_vit import VisionTransformer

st.set_page_config(
    page_title="PerceptiVIT",
    page_icon="🖼️",  # You can use an emoji or path to an image
    layout="centered"  # Options: 'centered', 'wide'
)

@st.cache_resource
def load_model(model_path="custom_vit_model.pth", 
               img_size=256, 
               embed_dim=192,  
               depth=4,       
               num_heads=4    
               ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VisionTransformer(
        img_size=img_size,
        patch_size=16,
        in_channels=3,
        num_classes=2,
        embed_dim=embed_dim,  
        depth=depth,          
        num_heads=num_heads,  
        mlp_ratio=4.0,
        dropout=0.1
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

model, device = load_model()

inference_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

st.title("PerceptiVIT - Real vs. Fake Image Classification with Vision Transformer")
st.write("Upload an image to see if it's predicted as **Real** or **Fake**.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    input_tensor = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        _, pred_class = torch.max(probs, 1)
        confidence = probs[0][pred_class].item()

    class_names = ["Fake", "Real"]
    st.write(f"**Prediction:** {class_names[pred_class]} "
             f"(Confidence: {confidence*100:.2f}%)")
