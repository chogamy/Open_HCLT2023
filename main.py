from argparse import ArgumentParser, Namespace

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar

from src.openlib.config import get_configurable_parameters
from src.openlib.data import get_datamodule
from src.openlib.models import get_model
from src.openlib.visualizer.visualizer import Visualizer


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=False,
        help="Path to a trainer config file",
    )
    parser.add_argument(
        "--known_cls_ratio",
        type=str,
        required=False,
        help="Path to a trainer config file",
    )
    parser.add_argument(
        "--sampler", type=str, required=False, help="Path to a overall config file"
    )
    parser.add_argument(
        "--config", type=str, required=False, help="Path to a overall config file"
    )
    parser.add_argument(
        "--model", type=str, required=False, help="Path to a model config file"
    )
    parser.add_argument(
        "--data", type=str, required=False, help="Path to a data config file"
    )
    parser.add_argument(
        "--trainer", type=str, required=False, help="Path to a trainer config file"
    )
    parser.add_argument(
        "--seed", type=str, required=False, help="Path to a trainer config file"
    )
    parser.add_argument(
        "--mode", type=str, required=False, help="Path to a trainer config file"
    )

    args = parser.parse_args()

    return args


def get_callbacks(callback_list):
    callbacks = []
    for callback in callback_list:
        callback_args = {
            callback["class_path"].split(".")[-1]: callback.get("init_args", None)
        }

        # TODO: too naive :(
        if "EarlyStopping" in callback_args:
            early_stopping = EarlyStopping(**callback_args["EarlyStopping"])
            callbacks.append(early_stopping)

        if "ModelCheckpoint" in callback_args:
            model_checkpoint = ModelCheckpoint(**callback_args["ModelCheckpoint"])
            callbacks.append(model_checkpoint)

        if "RichProgressBar" in callback_args:
            progress_bar = RichProgressBar()
            callbacks.append(progress_bar)

    return callbacks


def train():
    args = get_args()

    config = get_configurable_parameters(args=args)

    seed_everything(config.seed, workers=True)

    datamodule = get_datamodule(config)
    datamodule.prepare_data()

    config.model.init_args.num_classes = datamodule.num_classes
    model = get_model(config)
    callbacks = get_callbacks(config.trainer.pop("callbacks", []))

    trainer = Trainer(**config.trainer, callbacks=callbacks)

    if config.mode == "train":
        trainer.fit(model=model, datamodule=datamodule)
    if config.mode == "test":
        trainer.test(model=model, datamodule=datamodule)
    if config.mode == "visual":
        visualizer = Visualizer(model, datamodule, trainer)
        visualizer.run()


if __name__ == "__main__":
    train()
