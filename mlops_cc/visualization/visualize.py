from mlops_cc.models import model
import torch
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv



@click.command()
@click.argument("model_checkpoint", type=click.Path(exists=True))
def main(model_checkpoint):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Viz model data")
    print(model_checkpoint)

    mymodel = model.MyAwesomeModel(784, 10)
    mymodel.load_state_dict(torch.load(model_checkpoint))



if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
