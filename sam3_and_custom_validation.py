import gradio as gr
import torch
import numpy as np
import json
from PIL import Image
from transformers import Sam3Processor, Sam3Model
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

# --- 1. model loading
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading SAM 3 Model on {DEVICE}... (Please wait)")

try:
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3", trust_remote_code=True).to(DEVICE)
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"Error loading model: {e}")


# --- 2.core inference
def get_sam3_mask(image_pil, text_prompt, threshold=0.4):
    """Runs SAM3 and returns a boolean mask of ALL instances found"""
    width, height = image_pil.size
    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs, threshold=threshold, target_sizes=[(height, width)]
    )[0]

    masks = results["masks"].cpu().numpy()  # (N, H, W)
    scores = results["scores"].cpu().numpy()

    # Combine all valid instance masks into one binary mask
    combined_mask = np.zeros((height, width), dtype=bool)
    detected_count = 0
    confidences = []

    for i, score in enumerate(scores):
        if score > threshold:
            combined_mask = np.logical_or(combined_mask, masks[i])
            detected_count += 1
            confidences.append(score.item())

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return combined_mask, detected_count, avg_conf


# --- 3. UI TAB 1: Auto-Detection ---
def run_auto_detect(image, text_prompt):
    if image is None: return None, "Please upload an image."

    image_pil = Image.fromarray(image).convert("RGB")
    pred_mask, count, avg_conf = get_sam3_mask(image_pil, text_prompt)

    # Create Visual Overlay
    overlay = image.copy()
    overlay[pred_mask] = overlay[pred_mask] * 0.5 + np.array([0, 255, 0]) * 0.5  # Green tint

    report = {
        "Status": "Success",
        "Object": text_prompt,
        "Instances_Found": count,
        "Average_Confidence": f"{avg_conf:.4f}"
    }
    return overlay, json.dumps(report, indent=4)


# --- 4. UI TAB 2: Scientific Validation (Metrics)
def run_validation(image_editor_data, text_prompt):
    """
    Compares sam3 prediction vs human drawing (Ground Truth)
    """
    if image_editor_data is None: return "Error: No image provided"

    # Get the original image and the mask drawn by user
    # Gradio ImageEditor returns a dict: {'background': img, 'layers': [mask], 'composite': ...}
    image_arr = image_editor_data["background"]

    # The user's drawing is in 'layers'. we need to merge them into one binary mask.
    # Layers is often a list of RGBA images. so need the Alpha channel or non-black pixels.
    user_mask_layer = image_editor_data["layers"][0]  # assuming single layer drawing

    # convert user drawing to binary mask (Ground Truth)
    # check if any pixel is non-zero (drawn on)
    gt_mask = np.any(user_mask_layer[:, :, :3] > 0, axis=-1)

    if np.sum(gt_mask) == 0:
        return "Error: You didn't draw a mask! Please paint over the object first."

    # Run AI Inference
    image_pil = Image.fromarray(image_arr).convert("RGB")
    pred_mask, count, avg_conf = get_sam3_mask(image_pil, text_prompt)

    # --- CALCULATE METRICS ---
    # Flatten arrays for scikit-learn
    y_true = gt_mask.flatten()
    y_pred = pred_mask.flatten()

    # 1. IoU (Intersection over Union)
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    iou_score = intersection / union if union > 0 else 0.0

    # 2. Precision & Recall
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # format of results
    results = f"""
    ### Validation Report: '{text_prompt}'
    | Metric | Value | Description |
    | :--- | :--- | :--- |
    | **IoU Score** | **{iou_score:.4f}** | (Intersection Over Union) The most important accuracy metric. >0.5 is good, >0.8 is excellent. |
    | **Precision** | {precision:.4f} | How many detected pixels were actually the object? |
    | **Recall** | {recall:.4f} | Did SAM3 find ALL the object pixels you drew? |
    | **F1 Score** | {f1:.4f} | Balance between Precision and Recall. |
    | **AI Confidence** | {avg_conf:.4f} | How sure the SAM3 felt about its answer. |
    """
    return results


# --- 5. BUILD INTERFACE
with gr.Blocks(title="SAM 3 Validation Studio") as demo:
    gr.Markdown("# SAM 3: Object Segmentation & Metrics Studio")

    with gr.Tabs():
        # TAB 1: Quick Check
        with gr.TabItem("1. Auto-Detect"):
            with gr.Row():
                with gr.Column():
                    t1_input_img = gr.Image(label="Upload Image", sources=["upload", "clipboard"])
                    t1_text = gr.Textbox(label="What to find?", value="cyclist")
                    t1_btn = gr.Button("Segment Object", variant="primary")
                with gr.Column():
                    t1_output_img = gr.Image(label="AI Result")
                    t1_json = gr.JSON(label="Detection Data")

            t1_btn.click(run_auto_detect, inputs=[t1_input_img, t1_text], outputs=[t1_output_img, t1_json])

        # TAB 2: Validation (Metrics)
        with gr.TabItem("2. Validation (Calculate Confidence Score)"):
            gr.Markdown("""
            **How to use:**
            1. Upload an image in the editor below.
            2. Use the **Brush Tool** (top right of image) to color over the object (Ground Truth).
            3. Click 'Calculate Metrics' to compare SAM3 vs. Your Drawing.
            """)
            with gr.Row():
                with gr.Column():
                    # ImageEditor allows drawing masks
                    t2_editor = gr.ImageEditor(
                        label="Draw Ground Truth Here",
                        type="numpy",
                        brush=gr.Brush(colors=["#FF0000"], default_size=10),
                        eraser=gr.Eraser()
                    )
                    t2_text = gr.Textbox(label="Object Name", value="cyclist")
                    t2_btn = gr.Button("Calculate Accuracy (IoU)", variant="primary")
                with gr.Column():
                    t2_output_md = gr.Markdown("### Results will appear here...")

            t2_btn.click(run_validation, inputs=[t2_editor, t2_text], outputs=[t2_output_md])

    gr.HTML(
        """
        <div style="
            text-align: center; 
            margin-top: 25px; 
            padding: 15px; 
            background: linear-gradient(90deg, #f0f4ff 0%, #ffffff 50%, #f0f4ff 100%); 
            border-radius: 12px; 
            border: 1px solid #e0e6ed;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05); 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">

            <p style="margin: 0; font-size: 16px; color: #2d3748;">
                Designed & Developed by <b style="color: #4a5568;">Soban Hussan</b>
            </p>

            <p style="margin: 5px 0 0 0; font-size: 12px; color: #a0aec0; letter-spacing: 1px;">
                POWERED BY SAM 3 • COMPUTER VISION
            </p>
        </div>
        """
    )
# running ui
if __name__ == "__main__":
    demo.launch()