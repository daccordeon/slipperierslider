import numpy as np

def QFI_UB_bcTMS(g, G, t, nbar):
    """QFI upper bound with hiding unitary as TMS on modes b and c
    Args:
        g (float): Gain rate of mode a
        G (float): Loss rate of mode a
        t (float): Evolution time
        nbar (float): Initial average photon number in mode a
    Returns:
        float: QFI upper bound
    """         

    def mysqrt(x): return np.sqrt(x)

    aux0=(1.-((np.exp(((g-G)*t)))/(1.+(((1.-(np.exp(((g-G)*t))))*g)/(G-g)))))*nbar;
    aux1=(1.+(((1.-(np.exp(((g-G)*t))))*g)/(G-g)))*(nbar*((1.+aux0)*(((-1.+((np.exp(((g-G)*t)))+((G-g)*t)))**2))));
    aux2=((G-((np.exp(((g-G)*t)))*g))**-4.)*((((((np.exp((g*t)))*g)-((np.exp((G*t)))*G))**2))*aux1);
    aux3=(((np.exp(((2.*((g-G)*t))-(g*t))))*(g*(G*aux2)))/(g-G))/((np.exp((g*t)))-(np.exp((G*t))));
    aux4=((np.exp((g*t)))-(np.exp((G*t))))*((g-G)*(G*((((np.exp((g*t)))*g)-((np.exp((G*t)))*G))**-2.)));
    aux5=mysqrt(((np.exp(((g-G)*t)))/(1.+(((1.-(np.exp(((g-G)*t))))*g)/(G-g)))));
    aux6=mysqrt((1.-((np.exp(((g-G)*t)))/(1.+(((1.-(np.exp(((g-G)*t))))*g)/(G-g))))));
    aux7=(-1.+(mysqrt((1.+(((1.-(np.exp(((g-G)*t))))*g)/(G-g))))))/(1.+(mysqrt((1.+(((1.-(np.exp(((g-G)*t))))*g)/(G-g))))));
    aux8=(-1.+((np.exp(((g-G)*t)))+((G-g)*t)))*(G-((np.exp(((g-G)*t)))*(G+(g*((G-g)*t)))));
    aux9=(1.+(mysqrt((1.+(((1.-(np.exp(((g-G)*t))))*g)/(G-g))))))*((nbar**2)*aux8);
    aux10=(mysqrt((1.+(((1.-(np.exp(((g-G)*t))))*g)/(G-g)))))*(aux6*((mysqrt(aux7))*aux9));
    aux11=((G-((np.exp(((g-G)*t)))*g))**-2.)*((((np.exp((g*t)))*aux4)**-0.5)*(aux5*aux10));
    aux12=((1.+(mysqrt(((((np.exp(((g-G)*t)))*g)-G)/(g-G)))))**-0.5)*(((g-G)**-2.)*((((((np.exp(((g-G)*t)))*g)-G)/(g-G))**-0.5)*(G*aux11)));
    aux13=(np.exp(((g-G)*t)))*(g*(((-1.+(mysqrt(((((np.exp(((g-G)*t)))*g)-G)/(g-G)))))**-0.5)*aux12));
    aux14=1.+(((np.exp(((g-G)*t)))*nbar)/(1.+(((1.-(np.exp(((g-G)*t))))*g)/(G-g))));
    aux15=(-1.+(mysqrt((1.+(((1.-(np.exp(((g-G)*t))))*g)/(G-g))))))*((1.+(mysqrt((1.+(((1.-(np.exp(((g-G)*t))))*g)/(G-g))))))*(nbar*aux14));
    aux16=(((1.+(((1.-(np.exp(((g-G)*t))))*g)/(G-g)))*(1.+nbar))+aux15)*(((G-((np.exp(((g-G)*t)))*(G+(g*((G-g)*t)))))**2));
    aux17=((g*(((g-G)**-3.)*aux16))/(((np.exp(((g-G)*t)))*g)-G))/(1.+(mysqrt(((((np.exp(((g-G)*t)))*g)-G)/(g-G)))));
    aux18=aux3+((2.*aux13)+(aux17/(-1.+(mysqrt(((((np.exp(((g-G)*t)))*g)-G)/(g-G)))))));
    aux19=(-1.+(mysqrt((1.+(((1.-(np.exp(((g-G)*t))))*g)/(G-g))))))*((1.+(mysqrt((1.+(((1.-(np.exp(((g-G)*t))))*g)/(G-g))))))*nbar);
    aux20=(1.-((np.exp(((g-G)*t)))/(1.+(((1.-(np.exp(((g-G)*t))))*g)/(G-g)))))*nbar;
    output=(4.*aux18)/(aux19+((1.+(((1.-(np.exp(((g-G)*t))))*g)/(G-g)))*(1.+aux20)));
    return output