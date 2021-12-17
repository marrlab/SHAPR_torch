


def test_augmentation():

    assert obj == expected_obj
    assert img == expected_img


def test_data_generator(path, filenames):
    """
    test data_generator using a small data set from 'test_data' folder.
    """
    # can be a list of X_batch, Y_batch
    assert X_batch == expcted_X_batch
    assert Y_batch == expcted_X_batch


def test_data_generator_test_set(path, filenames):
    """
    test data_data_generator_test_set
    """
    assert img_out_2 == expcted_img_out_2


