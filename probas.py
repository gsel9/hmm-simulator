# Age range: 16 - 60 (53 y, Jerome)
TIME_RANGE = [
    range(0, 17),
    range(17, 37),
    range(37, 57),
    range(57, 77),
    range(77, 97),
    range(97, 137),
    range(137, 177),
    range(177, 321)
]



def p_init_state():
    pass


def p_s1_to_s2():
    pass


def p_s2_to_s3():
    pass


def p_s3_to_s4():
    pass


def p_s3_to_s2():
    pass


def p_s2_to_s1():
    pass


def transition_probas():
    """State transition probabilities."""
    # Age range x transition state
    P_trans = np.array(
        [
            [0.19910, 0.01665, 0.00251, 0.1771, 0.2262],
            [0.01202, 0.02526, 0.00025, 0.1550, 0.1079],
            [0.00731, 0.04176, 0.00014, 0.1448, 0.0811],
            [0.00573, 0.04201, 0.00017, 0.1520, 0.0739],
            [0.00537, 0.03467, 0.00048, 0.1553, 0.0670],
            [0.00537, 0.02938, 0.00096, 0.1664, 0.0830],
            [0.00429, 0.02495, 0.00128, 0.1933, 0.0959],
            [0.00395, 0.03383, 0.01325, 0.2348, 0.0582]
        ]
    )
    return P_trans


def init_screening_probas():
    """State probabilities at initial screening."""
    
    # Age range x risk state. Row stochastic.
    P_init = np.array(
        [
            [0.93020, 0.06693, 0.00263, 0.00024],
            [0.92937, 0.06228, 0.00821, 0.00014],
            [0.93384, 0.04945, 0.01654, 0.00017],
            [0.94875, 0.03574, 0.01528, 0.00023],
            [0.95348, 0.03226, 0.01400, 0.00026],
            [0.95543, 0.03309, 0.01132, 0.00016],
            [0.96316, 0.02806, 0.00847, 0.00031],
            [0.96032, 0.02793, 0.01134, 0.00041]
        ]
    )
    return P_init