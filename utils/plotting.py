import os


COLORS = ["#245c9c", "#d66b2a", "#2f7d55", "#8b3f92", "#5d5d5d"]


def _scale(values, lo, hi, size, invert=False):
    if hi <= lo:
        return [size / 2 for _ in values]
    coords = [(v - lo) / (hi - lo) * size for v in values]
    return [size - c if invert else c for c in coords]


def line_plot(path, series, title, xlabel, ylabel, width=760, height=460):
    margin = 58
    xs = [x for _, points in series for x, _ in points]
    ys = [y for _, points in series for _, y in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = (ymax - ymin) * 0.08 or 1.0
    ymin -= pad
    ymax += pad
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#222"/>',
        f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-family="Arial" font-size="13">{xlabel}</text>',
        f'<text transform="translate(16 {height/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">{ylabel}</text>',
    ]
    for idx, (name, points) in enumerate(series):
        px = _scale([x for x, _ in points], xmin, xmax, plot_w)
        py = _scale([y for _, y in points], ymin, ymax, plot_h, invert=True)
        coords = " ".join(f"{margin+x:.2f},{margin+y:.2f}" for x, y in zip(px, py))
        color = COLORS[idx % len(COLORS)]
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        parts.append(f'<text x="{width-margin-130}" y="{55+idx*18}" font-family="Arial" font-size="12" fill="{color}">{name}</text>')
    parts.append("</svg>")
    _write(path, "\n".join(parts))


def bar_plot(path, labels, values, title, ylabel, width=640, height=430):
    margin = 58
    ymax = max(values) * 1.15 if values else 1.0
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    bar_w = plot_w / max(1, len(values)) * 0.62
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#222"/>',
        f'<text transform="translate(16 {height/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">{ylabel}</text>',
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        center = margin + (i + 0.5) * plot_w / len(values)
        h = 0 if ymax == 0 else value / ymax * plot_h
        x = center - bar_w / 2
        y = height - margin - h
        color = COLORS[i % len(COLORS)]
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}"/>')
        parts.append(f'<text x="{center:.2f}" y="{height-margin+22}" text-anchor="middle" font-family="Arial" font-size="12">{label}</text>')
        parts.append(f'<text x="{center:.2f}" y="{y-6:.2f}" text-anchor="middle" font-family="Arial" font-size="12">{value:.3f}</text>')
    parts.append("</svg>")
    _write(path, "\n".join(parts))


def point_plot_with_ci(path, labels, values, ci_values, title, ylabel, width=720, height=440, xlabel="Uncertainty level"):
    margin = 64
    ymax = max([v + ci for v, ci in zip(values, ci_values)] + [1.0])
    ymin = min([v - ci for v, ci in zip(values, ci_values)] + [0.0])
    if ymax <= ymin:
        ymax = ymin + 1.0
    pad = (ymax - ymin) * 0.1
    ymin -= pad
    ymax += pad
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    def sx(i):
        if len(values) == 1:
            return margin + plot_w / 2
        return margin + i * plot_w / (len(values) - 1)

    def sy(v):
        return margin + (ymax - v) / (ymax - ymin) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#222"/>',
        f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-family="Arial" font-size="13">{xlabel}</text>',
        f'<text transform="translate(18 {height/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">{ylabel}</text>',
    ]
    coords = " ".join(f"{sx(i):.2f},{sy(v):.2f}" for i, v in enumerate(values))
    parts.append(f'<polyline points="{coords}" fill="none" stroke="{COLORS[0]}" stroke-width="2.4"/>')
    for i, (label, value, ci) in enumerate(zip(labels, values, ci_values)):
        x = sx(i)
        y = sy(value)
        y1 = sy(value - ci)
        y2 = sy(value + ci)
        parts.append(f'<line x1="{x:.2f}" y1="{y1:.2f}" x2="{x:.2f}" y2="{y2:.2f}" stroke="#333" stroke-width="1.5"/>')
        parts.append(f'<line x1="{x-6:.2f}" y1="{y1:.2f}" x2="{x+6:.2f}" y2="{y1:.2f}" stroke="#333" stroke-width="1.5"/>')
        parts.append(f'<line x1="{x-6:.2f}" y1="{y2:.2f}" x2="{x+6:.2f}" y2="{y2:.2f}" stroke="#333" stroke-width="1.5"/>')
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{COLORS[0]}"/>')
        parts.append(f'<text x="{x:.2f}" y="{height-margin+22}" text-anchor="middle" font-family="Arial" font-size="11">{label}</text>')
        parts.append(f'<text x="{x:.2f}" y="{y-10:.2f}" text-anchor="middle" font-family="Arial" font-size="11">{value:.3f}</text>')
    parts.append("</svg>")
    _write(path, "\n".join(parts))


def model_comparison_plot(path, labels, x_values, series, title, width=820, height=500):
    margin = 70
    all_y = []
    for item in series:
        all_y.extend(item["observed"])
        all_y.extend(item["linear"])
        all_y.extend(item["quadratic"])
    ymin = min(all_y + [0.0])
    ymax = max(all_y + [1.0])
    if ymax <= ymin:
        ymax = ymin + 1.0
    pad = (ymax - ymin) * 0.1
    ymin -= pad
    ymax += pad
    xmin = min(x_values)
    xmax = max(x_values)
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    def sx(x):
        if xmax <= xmin:
            return margin + plot_w / 2
        return margin + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y):
        return margin + (ymax - y) / (ymax - ymin) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#222"/>',
        f'<text x="{width/2}" y="{height-16}" text-anchor="middle" font-family="Arial" font-size="13">Uncertainty level</text>',
        f'<text transform="translate(20 {height/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">Metric value</text>',
    ]
    for i, label in enumerate(labels):
        parts.append(f'<text x="{sx(x_values[i]):.2f}" y="{height-margin+22}" text-anchor="middle" font-family="Arial" font-size="10">{label}</text>')
    for idx, item in enumerate(series):
        color = COLORS[idx % len(COLORS)]
        observed = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(x_values, item["observed"]))
        linear = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(x_values, item["linear"]))
        quadratic = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(x_values, item["quadratic"]))
        parts.append(f'<polyline points="{linear}" fill="none" stroke="{color}" stroke-width="1.8" stroke-dasharray="7 5"/>')
        parts.append(f'<polyline points="{quadratic}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        parts.append(f'<polyline points="{observed}" fill="none" stroke="{color}" stroke-width="1.2" opacity="0.45"/>')
        for x, y in zip(x_values, item["observed"]):
            parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4" fill="{color}"/>')
        y0 = 58 + idx * 34
        parts.append(f'<text x="{width-margin-170}" y="{y0}" font-family="Arial" font-size="12" fill="{color}">{item["name"]} observed</text>')
        parts.append(f'<line x1="{width-margin-170}" y1="{y0+8}" x2="{width-margin-138}" y2="{y0+8}" stroke="{color}" stroke-width="1.8" stroke-dasharray="7 5"/>')
        parts.append(f'<text x="{width-margin-132}" y="{y0+12}" font-family="Arial" font-size="11" fill="{color}">linear</text>')
        parts.append(f'<line x1="{width-margin-80}" y1="{y0+8}" x2="{width-margin-48}" y2="{y0+8}" stroke="{color}" stroke-width="2.4"/>')
        parts.append(f'<text x="{width-margin-42}" y="{y0+12}" font-family="Arial" font-size="11" fill="{color}">quad</text>')
    parts.append("</svg>")
    _write(path, "\n".join(parts))


def heatmap(path, counts, grid_size, title, width=460, height=500):
    margin = 42
    cell = min((width - 2 * margin) / grid_size, (height - 2 * margin - 24) / grid_size)
    max_count = max(counts.values()) if counts else 1
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="17" font-weight="700">{title}</text>',
    ]
    for y in range(grid_size):
        for x in range(grid_size):
            c = counts.get((x, y), 0)
            intensity = 0 if max_count == 0 else c / max_count
            red = int(245 - 185 * intensity)
            green = int(248 - 110 * intensity)
            blue = int(250 - 55 * intensity)
            parts.append(
                f'<rect x="{margin+x*cell:.2f}" y="{margin+24+y*cell:.2f}" width="{cell:.2f}" height="{cell:.2f}" '
                f'fill="rgb({red},{green},{blue})" stroke="#dddddd" stroke-width="0.5"/>'
            )
    parts.append("</svg>")
    _write(path, "\n".join(parts))


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
