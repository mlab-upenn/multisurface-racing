import numpy as np


def tire_forces_linear_model(alpha_r, alpha_f, kappa_r, kappa_f, Fzf=None, Fzr=None, Pkx1=None, Pky1=None, Cyr=None,
                             Cyf=None, Cxr=None, Cxf=None):
    """
    :param alpha_r: slip angle rear tire
    :param alpha_f: slip angle front tire
    :param kappa_r: slip ratio rear tire
    :param kappa_f: slip ratio front tire
    :param Fzf: downforce on front tire
    :param Fzr: downforce on rear tire
    :param Pkx1:
    :param Pky1:
    :param Cyr: lateral rear tire stiffness
    :param Cyf: lateral front tire stiffness
    :param Cxr: longitudinal rear tire stiffness
    :param Cxf: longitudinal front tire stiffness
    :return:
    """
    if Cyr is None:
        Cyr = Fzr * Pky1

    if Cyf is None:
        Cyf = Fzf * Pky1

    if Cxr is None:
        Cxr = Fzr * Pkx1

    if Cxf is None:
        Cxf = Fzf * Pkx1

    Fyf = Cyf * alpha_f
    Fyr = Cyr * alpha_r
    Fxf = Cxf * kappa_f
    Fxr = Cxr * kappa_r

    return Fyf, Fyr, Fxf, Fxr
