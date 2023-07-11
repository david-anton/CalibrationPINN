from parametricpinn.types import Module


def _freeze_model(model: Module) -> None:
    model.train(False)
    for parameters in model.parameters():
        parameters.requires_grad = False
