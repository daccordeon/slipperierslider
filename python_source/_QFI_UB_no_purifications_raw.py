def mysqrt(x): return np.sqrt((1.+0j)*x)

aux0=(g*(((g-G)**-3.)*(((G-((np.exp(((g-G)*t)))*(G+(g*((G-g)*t)))))**2))))/(((np.exp(((g-G)*t)))*g)-G);
aux1=(aux0/(1.+(mysqrt(((((np.exp(((g-G)*t)))*g)-G)/(g-G))))))/(-1.+(mysqrt(((((np.exp(((g-G)*t)))*g)-G)/(g-G)))));
aux2=(((((np.exp((g*t)))*g)-((np.exp((G*t)))*G))**2))*(((-1.+((np.exp(((g-G)*t)))+((G-g)*t)))**2));
aux3=(np.exp(((2.*((g-G)*t))-(g*t))))*(g*(G*(((G-((np.exp(((g-G)*t)))*g))**-4.)*aux2)));
aux4=(np.exp(((g-G)*t)))*(g*(((g-G)**-3.)*(((G-((np.exp(((g-G)*t)))*(G+(g*((G-g)*t)))))**2))));
aux5=(aux4/(1.+(((1.-(np.exp(((g-G)*t))))*g)/(G-g))))/(((np.exp(((g-G)*t)))*g)-G);
aux6=(aux5/(1.+(mysqrt(((((np.exp(((g-G)*t)))*g)-G)/(g-G))))))/(-1.+(mysqrt(((((np.exp(((g-G)*t)))*g)-G)/(g-G)))));
output=4.*(aux1+(nbar*(((aux3/(g-G))/((np.exp((g*t)))-(np.exp((G*t)))))+aux6)));
