def checkHealth():
    return "I'm still alive "


# import cv2
# import torch
# import numpy as np
# from PIL import Image
# import torchvision.transforms as transforms
# import torchvision

# def load_yolo_model():
#     # Load pre-trained YOLOv5 model
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#     return model

# def run_yolo_detection(model, image_path):
#     results = model(image_path)
#     results.show()

# def preprocess_image(image_path):
#     img = Image.open(image_path)
#     transform = transforms.ToTensor()
#     img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
#     return img_tensor

# def run_mask_rcnn(img_tensor, threshold=0.5):
#     model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
#     model.eval()
#     with torch.no_grad():
#         predictions = model(img_tensor)
#     masks = predictions[0]['masks'][predictions[0]['scores'] > threshold]
#     return masks

# def tile_pattern(camouflage, mask_shape):
#     camo_h, camo_w = camouflage.shape[:2]
#     mask_h, mask_w = mask_shape[:2]
#     tile_x = int(np.ceil(mask_w / camo_w))
#     tile_y = int(np.ceil(mask_h / camo_h))
#     tiled_camo = np.tile(camouflage, (tile_y, tile_x, 1))
#     return tiled_camo[:mask_h, :mask_w]

# def apply_camouflage(camo_img_path, original_image_path, output_path):
#     # Load YOLOv5 model
#     yolo_model = load_yolo_model()

#     # Preprocess the original image for Mask R-CNN
#     img_tensor = preprocess_image(original_image_path)

#     # Run Mask R-CNN to get human masks
#     human_masks = run_mask_rcnn(img_tensor)

#     # Load camouflage pattern and tile it to match the human mask size
#     camouflage = cv2.imread(camo_img_path)
#     tiled_camo = tile_pattern(camouflage, human_masks[0].shape)

#     # Apply the tiled camouflage pattern to the mask
#     masked_camo = cv2.bitwise_and(tiled_camo, tiled_camo, mask=human_masks[0][0].cpu().numpy().astype(np.uint8))

    
#     # Save the final camouflaged image
#     cv2.imwrite(f'./static/images/imprint/{output_path}', masked_camo)

#     return 1


import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torchvision
import matplotlib.pyplot as plt


def apply_camouflage(camo_img_path, original_image_path, output_path):

    # Step 1: Load YOLOv5 model
    model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Step 2: Run YOLOv5 detection on the original image
    results = model_yolo(original_image_path)
    # results.show()

    # Step 3: Load and preprocess the image for segmentation
    img = Image.open(original_image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)

    # Step 4: Load Mask R-CNN model for human detection and segmentation
    model_maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model_maskrcnn.eval()

    # Perform inference to get the mask for humans
    with torch.no_grad():
        predictions = model_maskrcnn(img_tensor)

    # Step 5: Extract masks with high confidence
    threshold = 0.5
    masks = predictions[0]['masks'][predictions[0]['scores'] > threshold]

    # Assuming a single person mask for simplicity
    human_mask = masks[0].squeeze().mul(255).byte().cpu().numpy()

    # Step 6: Load the camouflage pattern
    camouflage = cv2.imread(camo_img_path)

    # Function to tile the camouflage pattern over the mask
    def tile_pattern(camouflage, mask_shape):
        camo_h, camo_w = camouflage.shape[:2]
        mask_h, mask_w = mask_shape[:2]
        tile_x = int(np.ceil(mask_w / camo_w))
        tile_y = int(np.ceil(mask_h / camo_h))
        tiled_camo = np.tile(camouflage, (tile_y, tile_x, 1))
        return tiled_camo[:mask_h, :mask_w]


    # Load the camouflage pattern using the camo path
    camouflage_ori = cv2.imread(camo_img_path)

    # Resize the camouflage pattern to match the size of the original image
    camouflage = cv2.resize(camouflage_ori, (img.width, img.height))

    # Convert the human mask to grayscale if it's in color
    if len(human_mask.shape) == 3:  # If mask is in color (3 channels)
        human_mask = cv2.cvtColor(human_mask, cv2.COLOR_BGR2GRAY)

    # Ensure the mask is binary (0 and 255)
    _, human_mask = cv2.threshold(human_mask, 128, 255, cv2.THRESH_BINARY)

    # Apply the human mask to the camouflage pattern
    masked_camo = cv2.bitwise_and(camouflage, camouflage, mask=human_mask)

    # Create the inverse of the human mask
    human_mask_inv = cv2.bitwise_not(human_mask)  # Invert the human mask

    # Convert the original image to a NumPy array (in BGR format for OpenCV)
    image_np = np.array(img)

    # Ensure the original image is in BGR format (handling possible RGBA or RGB)
    if image_np.shape[2] == 4:  # RGBA to RGB conversion
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    elif image_np.shape[2] == 3:  # RGB to BGR conversion if using a PIL image
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Ensure that the mask and the original image are the same size
    if image_np.shape[:2] != human_mask.shape[:2]:
        human_mask = cv2.resize(human_mask, (image_np.shape[1], image_np.shape[0]))
        human_mask_inv = cv2.bitwise_not(human_mask)

    # Remove the human from the original image by applying the inverted human mask
    # This will black out the human in the original image
    background = cv2.bitwise_and(image_np, image_np, mask=human_mask_inv)

    # Combine the masked camouflage (human-shaped pattern) with the blacked-out background
    final_image = cv2.add(masked_camo, background)
    
 

    # Optionally, save the final image
    cv2.imwrite(f'./static/images/imprint/{output_path}', final_image)
    # cv2.imwrite(f'./static/images/detections/{output_path}', annotated_image)

    # Access the image with detections
    # YOLOv5 uses the 'ims' attribute to store annotated images
    results.render()
    annotated_image = results.ims[0]  # This will give you the annotated image

    # Save the image using OpenCV's imwrite
    cv2.imwrite(f'./static/images/detections/{output_path}', annotated_image)

    # results.save(save_dir='./static/images/detections/')

    return 1

