import cv2
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def load_trained_model(weights_path, num_classes=2):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    #Load weights
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    
    model.eval() 
    return model

def evaluate_video(model, video_path, output_path, confidence_threshold=0.5):
    device = torch.device('cpu')
    model.to(device)

    #Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    #Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    #Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        #BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #Convert to PyTorch tensor, normalize to 0.0-1.0
        img_tensor = torch.tensor(rgb_frame).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(device)

        #Run inference
        with torch.no_grad():
            predictions = model([img_tensor])[0]

        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()

        #Draw boxes 
        for box, score in zip(boxes, scores):
            if score > confidence_threshold:
                xmin, ymin, xmax, ymax = map(int, box)
                
                #Red rectangle
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
                
                #Confidence label
                label = f"Baseball: {score:.2f}"
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        out.write(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames.")

    cap.release()
    out.release()
    print(f"Annotated video saved here: {output_path}")

if __name__ == "__main__":
    model_weights = 'baseball_weights.pth'
    input_video = r'video_data\IMG_0173.mov' #Replace me
    output_video = 'annotated_pitch.mp4'     
    trained_model = load_trained_model(model_weights)
    evaluate_video(trained_model, input_video, output_video, confidence_threshold=0.5)