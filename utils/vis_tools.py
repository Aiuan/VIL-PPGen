import torch

def gray2jet(gray, vmax=1.0, vmin=0.0):
    def interpolate(val, y0, x0, y1, x1):
        return (val - x0) * (y1 - y0) / (x1 - x0) + y0

    def base(val):
        return (val > -0.75) * (val <= -0.25) * interpolate(val, 0.0, -0.75, 1.0, -0.25) \
            + (val > -0.25) * (val <= 0.25) * 1.0 \
            + (val > 0.25) * (val <= 0.75) * interpolate(val, 1.0, 0.25, 0.0, 0.75)

    def red(gray_norm):
        gray = gray_norm * 2 - 1
        return base(gray - 0.5)

    def green(gray_norm):
        gray = gray_norm * 2 - 1
        return base(gray)

    def blue(gray_norm):
        gray = gray_norm * 2 - 1
        return base(gray + 0.5)

    # gray(B, 1, W, H)

    # element in gray_norm 0-1
    gray_limit = gray.clamp(min=vmin, max=vmax)
    gray_norm = (gray_limit - vmin) / (vmax - vmin) + vmin

    r = red(gray_norm)
    g = green(gray_norm)
    b = blue(gray_norm)
    # rgb(B, 3, W, H)
    rgb = torch.cat((r, g, b), dim=1)
    return rgb
