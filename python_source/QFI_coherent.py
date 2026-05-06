import numpy as np

def QFI_coherent(g, G, t, alpha):
    """Calculate the Quantum Fisher Information for a coherent state.
    Args:
        g (float): Gain parameter.
        G (float): Loss parameter.
        t (float): Time.
        alpha (float): Coherent state amplitude.
    Returns:
        float: Quantum Fisher Information.
    """
    aux0=((-2.*(G*((2.*g)+G)))+(2.*(g*((g-G)*((g+G)*t)))))-((alpha**2)*(g*(((g-G)**3.)*(t**2))));
    aux1=(-2.*(g*((g-G)*(G*(((3.*g)+G)*t)))))+(g*((((g-G)**2))*((g+G)*(((g+((alpha**2)*g))-((alpha**2)*G))*(t**2)))));
    aux2=((np.exp(((g+(2.*G))*t)))*(G*aux0))+((np.exp((((2.*g)+G)*t)))*(((G**2)*((5.*g)+G))+aux1));
    aux3=(4.*(g*(G*((G-g)*t))))+(g*((((g-G)**2))*((((2.+(alpha**2))*g)-((alpha**2)*G))*(t**2))));
    aux4=(((np.exp((3.*(G*t))))*((G**2)*(g+G)))+aux2)-((np.exp((3.*(g*t))))*(g*((2.*(G**2))+aux3)));
    aux5=(4.*(((g-G)**-2.)*aux4))/((-2.*((np.exp((g*t)))*g))+((np.exp((G*t)))*(g+G)));
    output=(aux5/(((np.exp((G*t)))*G)-((np.exp((g*t)))*g)))/((np.exp((G*t)))-(np.exp((g*t))));
    return output