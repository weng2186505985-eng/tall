import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if isinstance(output, tuple):
            output = output[0]
            
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        target = output[0, target_class]
        target.backward()

        # weights: (B, C) -> average of gradients over spatial dimensions
        # self.gradients: (B, C, H, W) or (B, L, C)
        grads = self.gradients
        acts = self.activations
        
        if len(grads.shape) == 3: # Transformer tokens (B, L, C)
            # Reshape tokens to 2D if possible, otherwise treat as 1D sequence
            # For Swin-B 224, final tokens are often 7x7=49
            b, l, c = grads.shape
            side = int(np.sqrt(l))
            grads = grads.view(b, side, side, c).permute(0, 3, 1, 2)
            acts = acts.view(b, side, side, c).permute(0, 3, 1, 2)

        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        gradcam = torch.sum(weights * acts, dim=1, keepdim=True)
        gradcam = F.relu(gradcam)
        
        # Normalize
        gradcam = gradcam - gradcam.min()
        gradcam = gradcam / (gradcam.max() + 1e-7)
        
        return gradcam.detach().cpu().numpy()[0, 0]

def save_visualization(image, heatmap, output_path):
    # image: (H, W, 3) numpy array
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    visualization = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
    cv2.imwrite(output_path, visualization)

if __name__ == "__main__":
    print("Grad-CAM utils ready.")
