from models.pattern.classes import (
    CLASS_NAMES,
    ArmorFolder,
    PatternId,
    pattern_for_folder,
)


def test_class_names_are_hkust_six() -> None:
    assert CLASS_NAMES == ("1", "2", "3", "4", "S", "Q")
    assert len(PatternId) == 6


def test_blue_and_red_same_digit_share_id() -> None:
    assert pattern_for_folder(ArmorFolder.B1) is PatternId.DIGIT_1
    assert pattern_for_folder(ArmorFolder.R1) is PatternId.DIGIT_1
    assert pattern_for_folder(ArmorFolder.BS) is PatternId.SENTRY
    assert pattern_for_folder(ArmorFolder.R0) is PatternId.OUTPOST


def test_number_five_is_excluded() -> None:
    assert pattern_for_folder(ArmorFolder.B5) is None
    assert pattern_for_folder(ArmorFolder.R5) is None
