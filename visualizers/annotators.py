import supervision as sv
from config.settings import COLORS

class VideoAnnotator:
    def __init__(self, resolution: tuple[int, int]):
        self.thickness = sv.calculate_optimal_line_thickness(resolution)
        self.text_scale = sv.calculate_optimal_text_scale(resolution)

        self.human = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex([COLORS["team_1"], COLORS["goalkeeper"], COLORS["team_2"]]),
            thickness=self.thickness
        )
        
        self.ball = sv.CircleAnnotator(
            color=sv.Color.from_hex(COLORS["ball"]),
            thickness=self.thickness + 1
        )
        
        self.ball_carrying = sv.BoxCornerAnnotator(
            color=sv.Color.from_hex(COLORS["ball"]),
            thickness=self.thickness
        )
        
        self.speed_label = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex([COLORS["team_1"], COLORS["goalkeeper"]]),
            text_color=sv.Color.from_hex(COLORS["text"]),
            text_scale=self.text_scale / 2.5,
            text_thickness=self.thickness // 3,
            text_position=sv.Position.BOTTOM_CENTER
        )