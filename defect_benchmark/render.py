"""Fixed-frame coordinate renderer consolidated from PR #1."""
import math
from PIL import Image, ImageDraw

def render(coordinates, *, radius=1.0, center=(0.0, 0.0), size=448, margin=16, dot_radius=2):
    """Fixed physical frame, x right/y up. Reject invalid/out-of-wafer points."""
    if not math.isfinite(radius) or radius <= 0 or len(center) != 2 or not all(math.isfinite(v) for v in center):
        raise ValueError('Invalid wafer geometry')
    if size < 32 or margin < dot_radius + 2 or 2 * margin >= size or dot_radius < 1:
        raise ValueError('Invalid render dimensions')
    image = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(image)
    half = (size - 1) / 2
    scale = (half - margin) / radius
    draw.ellipse((margin, margin, size - 1 - margin, size - 1 - margin), outline='#808080', width=1)
    for point in coordinates:
        if len(point) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point):
            raise ValueError('Each coordinate must contain two finite numbers')
        x, y = point[0] - center[0], point[1] - center[1]
        if math.hypot(x, y) > radius * (1 + 1e-12):
            raise ValueError('Coordinate outside wafer; check center/radius/units')
        px, py = round(half + x * scale), round(half - y * scale)
        draw.ellipse((px-dot_radius, py-dot_radius, px+dot_radius, py+dot_radius), fill='black')
    return image

