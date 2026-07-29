from trainer import TrainingProgressCallback


def test_training_progress_callback_implements_trainer_lifecycle() -> None:
    callback = TrainingProgressCallback()

    assert callable(callback.on_init_end)
    assert callable(callback.on_train_begin)
    assert callable(callback.on_step_end)
    assert callable(callback.on_log)
