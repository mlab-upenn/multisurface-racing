from datetime import datetime
import torch

def save_GP_enemble_model(model, filepath=''):
    """Save GP model to a file.
    """
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")

    torch.save(model.gp_model.state_dict(), filepath + 'gp' + dt_string + '.pth')
    torch.save(model.gp_likelihood.state_dict(), filepath + 'gp_likelihood' + dt_string + '.pth')