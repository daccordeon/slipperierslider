import numpy as np

def QFI_asymmetric_TMSV(eta, sigma, r, s):
    """ Quantum Fisher Information for Two-Mode Squeezed Vacuum (TMSV) state in an asymmetric lossy channel.
    Args:
        eta (float): Transmissivity of the channel (flipped from stochastic paper which uses 1 - eta).
        sigma (float): Noise parameter of the asymmetric additive noise signal.
        r (float): Two-mode squeezing parameter of the TMSV state.
        s (float): Secondary single-mode squeezing parameter.
    Returns:
        float: Quantum Fisher Information.
    """
    aux0=((np.exp((2.*r)))-(np.exp((-2.*r))))*(((np.exp((4.*r)))-(np.exp((-4.*r))))*((((-1.+eta)**2))*eta));
    aux1=(np.exp((8.*s)))*(((((np.exp((-2.*r)))+(np.exp((2.*r))))**2))*((eta**2)*(sigma**2)));
    aux2=0.5*(((np.exp((-4.*r)))+(np.exp((4.*r))))*(-1.+(eta+(-6.*(sigma**2)))));
    aux3=((np.exp((-2.*r)))+(np.exp((2.*r))))*((-1.+eta)*(eta*((1.+((-10.*(sigma**2))+aux2))-eta)));
    aux4=((np.exp((-8.*r)))+(np.exp((8.*r))))*((-1.+eta)*(-1.+(eta+(-2.*(sigma**2)))));
    aux5=((np.exp((-4.*r)))+(np.exp((4.*r))))*(1.+((eta**2)+(-4.*((-1.+eta)*(sigma**2)))));
    aux6=(-2.*eta)+((5.*(eta**2))+((14.*((-1.+eta)*(sigma**2)))+((-0.5*aux4)+(-2.*aux5))));
    aux7=(8.*aux1)+((2.*((np.exp((6.*s)))*aux3))+((np.exp((4.*s)))*((-1.+eta)*(5.+aux6))));
    aux8=((np.exp((-4.*r)))+(np.exp((4.*r))))*((-1.+eta)*(-1.+(eta+(-2.*(sigma**2)))));
    aux9=(0.5*(((np.exp((2.*s)))-(np.exp((-2.*s))))*(sigma**2)))+(0.5*(((np.exp((-2.*s)))+(np.exp((2.*s))))*((1.+(sigma**2))-eta)));
    aux10=((-1.+eta)*(1.+((3.*eta)+(-2.*(sigma**2)))))+(2.*(((np.exp((-2.*r)))+(np.exp((2.*r))))*(eta*aux9)));
    aux11=((np.exp((-2.*r)))+(np.exp((2.*r))))*(((np.exp((2.*s)))-(np.exp((-2.*s))))*(eta*(sigma**2)));
    aux12=((np.exp((-2.*r)))+(np.exp((2.*r))))*(((np.exp((-2.*s)))+(np.exp((2.*s))))*eta);
    aux13=((0.5*(((np.exp((-4.*r)))+(np.exp((4.*r))))*(-1.+eta)))-aux12)*((-1.+eta)-(sigma**2));
    aux14=3.+((-2.*eta)+((3.*(eta**2))+((-3.*((-1.+eta)*(sigma**2)))+(aux11+aux13))));
    output=(((np.exp((-4.*s)))*(((np.exp((2.*s)))*aux0)+aux7))/((0.5*aux8)+aux10))/aux14;
    return output