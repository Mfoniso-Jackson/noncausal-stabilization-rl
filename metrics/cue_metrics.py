def cue_zone_cells(cues, grid_size, radius=1):
    cells = set()
    for cx, cy in cues:
        for x in range(grid_size):
            for y in range(grid_size):
                if abs(x - cx) + abs(y - cy) <= radius:
                    cells.add((x, y))
    return cells


def cue_dependence_score(visits, cues, grid_size, radius=1):
    if not visits or not cues:
        return 0.0
    zone = cue_zone_cells(cues, grid_size, radius)
    observed = sum(1 for pos in visits if pos in zone) / len(visits)
    expected = len(zone) / float(grid_size * grid_size)
    if expected >= 1.0:
        return 0.0
    return max(0.0, (observed - expected) / (1.0 - expected))


def stabilization_persistence_index(before_cds, after_cds):
    if before_cds <= 1e-9:
        return 0.0
    return max(0.0, min(1.0, after_cds / before_cds))

