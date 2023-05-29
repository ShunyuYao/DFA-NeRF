import torch
import point_render

"""
    Define a renderer tool.
    For each time, input a proj_points (B, N_V, 3) and threshold (float), output an depth_image (B, h, w, 1) and weight image (B, h, w, 1):
        depth_image / weight_image = render_image
    
    image_height: the height of image
    image_width: the width of image
    visible_patch_width: the width of patch which define if a point is visible.
    color_path_width: the width of patch which define the region a point affects.
    epsilon: avoid the divisor == 0
    
    proj_points: (B, N_V, 3). N_V is the number of points. (x, y) must on the image, z must under 0.0 (z smaller, more closer to camera).
    threshold: the threshold which decides if the point is visible.
"""

class Render_Point(torch.autograd.Function):
    @staticmethod
    def forward(ctx, proj_points, proj_color, Imweights, mask, image_height, image_width, visible_patch_width, color_path_width, epsilon, threshold):
        pixel_index, is_visible, weight_points, depth_image, color_image, Imweights_image, weight_image = \
            point_render.forward(proj_points, proj_color, Imweights, mask, image_height, image_width, visible_patch_width, color_path_width, epsilon, threshold)
        ctx.save_for_backward(proj_points, proj_color, Imweights, mask, pixel_index, is_visible, weight_points)
        return depth_image, color_image, Imweights_image, weight_image, is_visible

    @staticmethod
    def backward(ctx, grad_depth_image, grad_color_image, grad_Imweights_image, grad_weight_image, _):
        proj_points, proj_color, Imweights, mask, pixel_index, is_visible, weight_points = ctx.saved_tensors
        output = \
            point_render.backward(grad_depth_image, grad_color_image, grad_Imweights_image, grad_weight_image, proj_points, proj_color, Imweights, mask, pixel_index, is_visible, weight_points)[0]
        grad_proj_points = output[:, :, :3]
        grad_Imweights = output[:, :, 3:]
        return grad_proj_points, grad_Imweights, None, None, None, None, None, None, None, None

class Render(torch.nn.Module):
    def __init__(self, image_height=576, image_width=640, visible_patch_width=5, color_path_width=7, epsilon=1e-5):
        super(Render, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.visible_patch_width = visible_patch_width
        self.color_path_width = color_path_width
        self.epsilon = epsilon # avoid division by zero

    def forward(self, proj_points, proj_color, Imweights, mask, threshold):
        mask = mask.int()
        return Render_Point.apply(proj_points, proj_color, Imweights, mask, self.image_height, self.image_width,
                                    self.visible_patch_width, self.color_path_width, self.epsilon, threshold)
