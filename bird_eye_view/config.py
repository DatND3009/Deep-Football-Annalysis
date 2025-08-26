class FootballPitchConfig:
    def __init__(self):
        self.width = 7000
        self.length = 12000
        self.penalty_box_width = 4100
        self.penalty_box_length = 2015
        self.goal_box_width = 1832
        self.goal_box_length = 550
        self.centre_circle_radius = 915
        self.penalty_spot_distance = 1100  # by x-axis

    def get_vertices(self):
        return [
            (0, 0), #1
            (0, (self.width - self.penalty_box_width) / 2), #2
            (0, (self.width - self.goal_box_width) / 2), #3
            (0, (self.width + self.goal_box_width) / 2), #4
            (0, (self.width + self.penalty_box_width) / 2), #5
            (0, self.width), #6
            (
                self.goal_box_length,
                (self.width - self.goal_box_width) / 2
            ), #7
            (
                self.goal_box_length,
                (self.width + self.goal_box_width) / 2
            ), #8
            (self.penalty_spot_distance, self.width / 2), #9
            (
                self.penalty_box_length,
                (self.width - self.penalty_box_width) / 2
            ), #10
            (
                self.penalty_box_length,
                (self.width - self.goal_box_width) / 2
            ), #11
            (
                self.penalty_box_length,
                (self.width + self.goal_box_width) / 2
            ), #12
            (
                self.penalty_box_length,
                (self.width + self.penalty_box_width) / 2
            ), #13
            (self.length / 2, 0), #14
            (
                self.length / 2,
                self.width / 2 - self.centre_circle_radius
            ), #15
            (
                self.length / 2,
                self.width / 2 + self.centre_circle_radius
            ), #16
            (self.length / 2, self.width), #17
            (
                self.length - self.penalty_box_length,
                (self.width - self.penalty_box_width) / 2
            ), #18
            (
                self.length - self.penalty_box_length,
                (self.width - self.goal_box_width) / 2
            ), #19
            (
                self.length - self.penalty_box_length,
                (self.width + self.goal_box_width) / 2
            ), #20
            (
                self.length - self.penalty_box_length,
                (self.width + self.penalty_box_width) / 2
            ), #21
            (
                self.length - self.centre_circle_radius,
                self.width / 2
            ), #22
            (
                self.length - self.goal_box_length,
                (self.width - self.goal_box_width) / 2
            ), #23
            (
                self.length - self.goal_box_length,
                (self.width + self.goal_box_width) / 2
            ), #24
            (self.length, 0), #25
            (
                self.length,
                (self.width - self.penalty_box_width) / 2
            ), #26
            (
                self.length,
                (self.width - self.goal_box_width) / 2
            ), #27
            (
                self.length,
                (self.width + self.goal_box_width) / 2
            ), #28
            (
                self.length,
                (self.width + self.penalty_box_width) / 2
            ), #29
            (self.length, self.width), #30
            (
                self.length / 2 - self.centre_circle_radius,
                self.width / 2
            ), #31
            (
                self.length / 2 + self.centre_circle_radius,
                self.width / 2
            ) #32
        ]

    def get_edges(self):
        return [
            (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
            (7, 8), (10, 11), (11, 12), (12, 13),
            (14, 15), (15, 16), (16, 17),
            (18, 19), (19, 20), (20, 21), (23, 24),
            (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
            (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
            (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30)
        ]