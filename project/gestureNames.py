from .gestureActions import GestureActions


class GestureNames:
    FIRSTGROUP = "FirstGroup"
    SECONDGROUP = "SecondGroup"
    THIRDGROUP = "ThirdGroup"
    FIRSTLETTER = "FirstLetter"
    SECONDLETTER = "SecondLetter"
    THIRDLETTER = "ThirdLetter"
    CLICK = "Click"
    NOTHING = "Nothing"
    BACK = "Back"
    SCROLL = "Scroll"

    gestureNames = [
        FIRSTGROUP,
        SECONDGROUP,
        THIRDGROUP,
        FIRSTLETTER,
        SECONDLETTER,
        THIRDLETTER,
        CLICK,
        NOTHING,
        BACK,
        SCROLL, 
    ]

    gestureColors = {
        FIRSTGROUP: (0, 60, 255),
        SECONDGROUP: (0, 131, 255),
        THIRDGROUP: (0, 220, 255),
        FIRSTLETTER: (252, 60, 0),
        SECONDLETTER: (255, 144, 0),
        THIRDLETTER: (255, 210, 0),
        CLICK: (255, 0, 0),
        BACK: (0, 0, 255),
        NOTHING: (255, 255, 255),
        SCROLL: (0, 0, 0),
    }

    gestureActionsSingle = {
        NOTHING: GestureActions.cursor,
    }

    gestureActionsMultiple = {
        f"{SCROLL}|{SCROLL}": GestureActions.scroll,
        f"{FIRSTGROUP}|{NOTHING}": GestureActions.left_first_group,
        f"{SECONDGROUP}|{NOTHING}": GestureActions.left_second_group,
        f"{THIRDGROUP}|{NOTHING}": GestureActions.left_third_group,
        f"{NOTHING}|{FIRSTGROUP}": GestureActions.right_first_group,
        f"{NOTHING}|{SECONDGROUP}": GestureActions.right_second_group,
        f"{NOTHING}|{THIRDGROUP}": GestureActions.right_third_group,
    }
