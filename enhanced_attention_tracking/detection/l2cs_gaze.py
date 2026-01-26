import logging
import numpy as np
import cv2
import sys

# Configure logging
logger = logging.getLogger(__name__)

# Check for PyTorch dependencies
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not installed. L2CS Deep Learning Gaze model will not be available.")
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    class L2CS(nn.Module):
        """
        L2CS-Net Architecture for robust Gaze Estimation.
        Predicts Pitch and Yaw using separate classification heads with Softmax expectation.
        """
        def __init__(self, num_bins=90, pretrained_backbone=True):
            super(L2CS, self).__init__()
            # Use ResNet50 backbone (standard for L2CS)
            # Weights=None for the user to provide fine-tuned gaze weights
            self.backbone = models.resnet50(pretrained=pretrained_backbone)
            
            num_features = self.backbone.fc.in_features
            
            # Replace original classification head
            self.backbone.fc = nn.Sequential()
            
            # Separate heads for Yaw and Pitch
            self.fc_yaw = nn.Linear(num_features, num_bins)
            self.fc_pitch = nn.Linear(num_features, num_bins)
            
        def forward(self, x):
            # Feature extraction
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)

            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)

            # Heads
            yaw = self.fc_yaw(x)
            pitch = self.fc_pitch(x)
            return yaw, pitch

class L2CSGazeEstimator:
    """
    Wrapper for L2CS Deep Learning Gaze Model.
    Provides precise, angle-robust gaze estimation.
    """
    def __init__(self, model_path, device=None):
        self.active = False
        if not TORCH_AVAILABLE:
            return
            
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        logger.info(f"Initializing L2CS Gaze Model on {self.device}...")
        
        # Initialize Architecture
        # Pretrained backbone=False because we load full weights next
        self.model = L2CS(num_bins=90, pretrained_backbone=False)
        
        # Load Weights
        try:
            # Map location handles CPU/GPU mismatch
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle potential state dict key mismatch
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # Remap keys to match our L2CS architecture
            # Our model uses 'backbone.' prefix for ResNet, and 'fc_yaw'/'fc_pitch' heads
            new_state_dict = {}
            for k, v in state_dict.items():
                # Remove DDP prefix if present
                if k.startswith("module."):
                    k = k[7:]
                
                # Map specific heads
                if k.startswith("fc_yaw_gaze"):
                    new_key = k.replace("fc_yaw_gaze", "fc_yaw")
                elif k.startswith("fc_pitch_gaze"):
                    new_key = k.replace("fc_pitch_gaze", "fc_pitch")
                elif k.startswith("fc_finetune"):
                    continue # Skip unknown head
                else:
                    # Assume it belongs to ResNet backbone
                    new_key = f"backbone.{k}"
                
                new_state_dict[new_key] = v
                
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            self.active = True
            logger.info("L2CS Deep Learning Model loaded successfully with key mapping.")
        except FileNotFoundError:
            logger.warning(f"L2CS weights not found at {model_path}. Using fallback methods.")
        except Exception as e:
            logger.error(f"Failed to load L2CS weights: {e}")
            
        # Transformations (Standard ResNet ImageNet stats)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(448), # L2CS standard input
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Output bins tensors
        self.idx_tensor = torch.arange(90).float().to(self.device)
        self.softmax = nn.Softmax(dim=1)
        
    def estimate_gaze(self, frame, face_bbox):
        """
        Estimate gaze angles (Pitch, Yaw) from frame and face bounding box.
        
        Args:
            frame: Numpy array (RGB)
            face_bbox: [x_min, y_min, x_max, y_max] pixels
            
        Returns:
            yaw (radians), pitch (radians)
        """
        if not self.active:
            return None, None
            
        try:
            h, w, c = frame.shape
            x1, y1, x2, y2 = map(int, face_bbox)
            
            # Padding context (critical for accuracy)
            # L2CS usually likes ~1.2x padding
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = int(bw * 0.1), int(bh * 0.1)
            
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            
            face_img = frame[y1:y2, x1:x2]
            
            # Safety check
            if face_img.size == 0 or face_img.shape[0] < 10 or face_img.shape[1] < 10:
                return None, None
                
            # Preprocess
            img_tensor = self.transform(face_img).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                pitch_out, yaw_out = self.model(img_tensor)
                
                # Soft Argmax expectation
                pitch_prob = self.softmax(pitch_out)
                yaw_prob = self.softmax(yaw_out)
                
                # Convert bins to degrees (4 degrees per bin, centered at 180?)
                # L2CS specific: (sum(prob * idx) * 4) - 180
                pitch_deg = torch.sum(pitch_prob.data[0] * self.idx_tensor) * 4 - 180
                yaw_deg = torch.sum(yaw_prob.data[0] * self.idx_tensor) * 4 - 180
                
                # Convert to radians
                pitch_rad = pitch_deg.item() * np.pi / 180.0
                yaw_rad = yaw_deg.item() * np.pi / 180.0
                
                return yaw_rad, pitch_rad
                
        except Exception as e:
            logger.error(f"L2CS Inference error: {e}")
            return None, None
