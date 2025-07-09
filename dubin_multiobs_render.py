import numpy as np
from PIL import Image, ImageDraw
import math

def draw_comet(draw, base_center, angle_rad, length, width, fill_color="blue"):
    angle_rad = -angle_rad
    radius = width / 2
    tip_x = base_center[0] + length * math.cos(angle_rad)
    tip_y = base_center[1] + length * math.sin(angle_rad)
    perp_angle = angle_rad + math.pi / 2
    p1_x = base_center[0] + radius * math.cos(perp_angle)
    p1_y = base_center[1] + radius * math.sin(perp_angle)
    p2_x = base_center[0] - radius * math.cos(perp_angle)
    p2_y = base_center[1] - radius * math.sin(perp_angle)
    draw.polygon([(tip_x, tip_y), (p1_x, p1_y), (p2_x, p2_y)], fill=fill_color)
    start_angle_deg = math.degrees(angle_rad) + 90
    end_angle_deg = math.degrees(angle_rad) - 90
    bounding_box = [
        (base_center[0] - radius, base_center[1] - radius),
        (base_center[0] + radius, base_center[1] + radius)
    ]
    draw.pieslice(bounding_box, start=start_angle_deg, end=end_angle_deg, fill=fill_color)


def state_to_image_pil_hq(s, args):
    """
    Generates a high-quality, anti-aliased image from a state.
    """
    # ✅ 1. Use a scale factor for anti-aliasing
    scale = 4
    size = args.size
    h_size = (size[0] * scale, size[1] * scale)
    
    img = Image.new('RGB', h_size, 'white')
    draw = ImageDraw.Draw(img)

    def to_pixel(coord):
        x, y = coord
        # We now map to the high-resolution canvas
        px = int((x + args.x_max) / (2 * args.x_max) * h_size[0])
        py = int((-y + args.y_max) / (2 * args.y_max) * h_size[1])
        return (px, py)

    # --- Draw the circle ---
    for obs_x, obs_y, obs_r in zip(args.obs_x, args.obs_y, args.obs_r):
        center_px = to_pixel((obs_x, obs_y))
        radius_px = (obs_r / (2 * args.x_max)) * h_size[0]
        draw.ellipse(
            [(center_px[0] - radius_px, center_px[1] - radius_px),
         (center_px[0] + radius_px, center_px[1] + radius_px)],
        outline='red',
        width=2 * scale  # ✅ Use a thin line, scaled up
    )

    # --- Draw the arrow ---
    agent_pos_world = (s[0].item(), s[1].item())
    agent_pos_pixel = to_pixel(agent_pos_world)
    agent_angle_rad = s[2].item()
    
    # Use the comet function, but with scaled length and width
    draw_comet(draw,
               base_center=agent_pos_pixel,
               angle_rad=agent_angle_rad,
               length=17.5 * scale,  # ✅ Scale the comet size
               width=10 * scale,
               fill_color='blue')

    # ✅ 2. Resize the image down to the original size with a high-quality filter
    img = img.resize(size, Image.Resampling.LANCZOS)

    return np.array(img)
