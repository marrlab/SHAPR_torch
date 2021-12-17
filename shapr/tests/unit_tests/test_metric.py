



def test_dice_coef():
    """
    unit test for dice_coef()
    """
    assert dice_coef() == expected_dice_coef


def test_dice_coef_loss():
    """
    unit test for dice_coef_loss()
    """
    assert dice_coef_loss() == expected_dice_coef_loss


def test_dice_crossentropy_loss():
    """
    unit test for dice_coef_loss()
    """
    assert dice_dice_crossentropy_loss() == expected_dice_crossentropy_loss
