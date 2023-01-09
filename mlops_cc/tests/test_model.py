from mlops_cc.models import model
import torch
import pytest


@pytest.mark.parametrize("batch_size", [64, 40, 12])
def test_model(batch_size):

    model_checkpoint = "models/trained_modelV2.pt"
    mymodel = model.MyAwesomeModel(784, 10)
    mymodel.load_state_dict(torch.load(model_checkpoint))

    in_data = torch.randn(batch_size,1,28,28)
    in_data_tr = in_data.view(in_data.shape[0], -1)

    out_data = mymodel(in_data_tr)

    assert out_data.shape == (in_data.shape[0], 10)

    # assert eval(model_output(model, test_input)) == expected



def test_error_wrong_shape():

    mymodel = model.MyAwesomeModel(784, 10)

    with pytest.raises(ValueError, match='Expected input to a 2D tensor'):
        mymodel(torch.randn(1,2,3,5))