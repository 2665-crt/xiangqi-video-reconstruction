from .chess import board_signature
from .schemas import Piece, Position, Square


class VisualRecovery:
    def __init__(self, reader):
        self.reader = reader
        self.check_count = 0
        self.reset()

    def reset(self):
        self.requested = False
        self.next_check_s = 0
        self.attempts = 0
        self.previous = None
        self.evidence = []

    def request(self, time_s):
        if not self.requested:
            self.requested = True
            self.next_check_s = time_s

    def observe(self, time_s, image, reference, cells):
        if not self.requested or time_s < self.next_check_s or len(cells) < 85:
            return None
        self.attempts += 1
        self.check_count += 1
        self.next_check_s = time_s + (1 if self.attempts < 3 else 10)
        labels = {}
        for square, reading in self.reader(image).items():
            probabilities = reading['class_probabilities']
            if not probabilities:
                continue
            label = max(probabilities, key=probabilities.get)
            color, _ = label.split('_', 1)
            if reading['confidence'] >= .75 and probabilities[label] >= .95 and cells.get(square) == color:
                labels[square] = label
        sample = (time_s, image, dict(cells), labels)
        previous = self.previous
        self.previous = sample
        if previous is None or time_s - previous[0] > 3:
            if all(color is None or s in labels for s, color in cells.items()):
                self.next_check_s = time_s + 1
            return None
        old_cells, old_labels = previous[2:]
        if (any(color != old_cells[s] for s, color in cells.items() if s in old_cells)
                or any(label != old_labels[s] for s, label in labels.items() if s in old_labels)):
            return None
        combined = old_cells | cells
        combined_labels = old_labels | labels
        if len(combined) != 90:
            return None
        known = {(p.square.file, p.square.rank): f'{p.color}_{p.kind}' for p in reference.pieces}
        for square, color in combined.items():
            if color is None:
                if square in known and (square not in cells or square not in old_cells):
                    return None
                continue
            label = combined_labels.get(square)
            if label is None:
                return None
            if label != known.get(square) and (labels.get(square) != label or old_labels.get(square) != label):
                return None
        position = Position(pieces=[Piece(id=f'{label}_{f}_{r}', color=label.split('_', 1)[0],
            kind=label.split('_', 1)[1], square=Square(file=f, rank=r))
            for (f, r), label in combined_labels.items() if combined.get((f, r)) is not None],
            side_to_move=reference.side_to_move)
        if board_signature(position) == board_signature(reference):
            self.reset()
            return None
        self.evidence = [(previous[0], previous[1]), (time_s, image)]
        self.requested = False
        return position
