
#import unittest
import matplotlib.pyplot as plt
import numpy as np

import BubbleModels as bm

avogadro = 6.02214e23 # Avogadro-Konstante [Teilchenzahl / mol]

def test_GilmoreEick():
    t, R, R_dot, pg, T = bm.GilmoreEick(500e-6, 0, 50e-6,
                                        0., 100e-6, 0.01e-6)

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(t, R, '.')
    plt.ylabel('$R$ [m]')
    plt.subplot(4, 1, 2)
    plt.plot(t, R_dot, '.-')
    plt.ylabel('$\dot R$ [m/s]')
    plt.subplot(4, 1, 3)
    plt.plot(t, pg, '.-')
    plt.ylabel('$p_g$ [Pa]')
    plt.subplot(4, 1, 4)
    plt.plot(T, '.-')
    plt.ylabel('$T$ [K]')
    plt.tight_layout()
    plt.show()
        
def test_Gilmore_ode():
    t, R, R_dot = bm.Gilmore_ode(500e-6, 0, 50e-6,
                                 0., 100., 0.010)

    t = t[0:-1]
    R = R[0:-1]
    R_dot = R_dot[0:-1]
    
    plt.figure()
    plt.title("Gilmore")
    plt.subplot(2, 1, 1)
    plt.plot(t / 1e-6, R / 1e-6, '.')
    plt.ylabel('$R$ [um]')
    plt.subplot(2, 1, 2)
    plt.plot(t / 1e-6, R_dot, '.-')
    plt.ylabel('$\dot R$ [m/s]')
    plt.xlabel('$t$ [us]')
    plt.tight_layout()
    plt.show()

def test_GilmoreEick_ode():
    t, R, R_dot, pg, T = bm.GilmoreEick_ode(500e-6, 0, 50e-6,
                                            0, 300, 0.010)
    
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(t, R, '.')
    plt.ylabel('$R$ [m]')
    plt.subplot(4, 1, 2)
    plt.plot(t, R_dot, '.-')
    plt.ylabel('$\dot R$ [m/s]')
    plt.subplot(4, 1, 3)
    plt.plot(t, pg, '.-')
    plt.ylabel('$p_g$ [Pa]')
    plt.subplot(4, 1, 4)
    plt.plot(T, '.-')
    plt.ylabel('$T$ [K]')
    plt.tight_layout()
    plt.show()

def test_Toegel_ode():
    print("Toegel")
    print("======")
    t, R, R_dot, N, T, p_ext, p_b, R_equ = bm.Toegel_ode(350e-6, 1e-9, 40.0,
                                                  12e-6,
                                                  0e-6, 150e-6, 10e-9)

    t = t[0:-1]
    R = R[0:-1]
    R_dot = R_dot[0:-1]
    N = N[0:-1]
    T = T[0:-1]
    p_ext = p_ext[0:-1]
    p_b = p_b[0:-1]
    R_equ = R_equ[0:-1]

    # Ergebnis abspeichern
    filename = 'Toegel_20C_O2.dat'
    daten = np.hstack((t.reshape(-1, 1) / 1e-6,
                       R.reshape(-1, 1) / 1e-6,
                       N.reshape(-1, 1)))
    #np.savetxt(filename, daten, delimiter='\t')
    
    plt.figure()
    nplots = 7
    plt.subplot(nplots, 1, 1)
    plt.title("Toegel, $p_{\mathrm{ac}} = 0\,$bar, $T = 20\,$C")
    plt.plot(t / 1e-6, R / 1e-6, '-')
    #plt.plot(t / 1e-6, R, '-')
    plt.ylabel('$R$ [um]')
    plt.subplot(nplots, 1, 2)
    plt.plot(t / 1e-6, R_dot, '-')
    #plt.plot(t / 1e-6, R_dot, '-')
    plt.ylabel('$\dot R$ [m/s]')
    plt.subplot(nplots, 1, 3)
    plt.plot(t / 1e-6, N * avogadro, '-')
    #plt.plot(t / 1e-6, N, '-')
    plt.yscale('log')
    plt.ylabel('Wasserdampf-Stoffmenge $N$')
    plt.subplot(nplots, 1, 4)
    plt.plot(t / 1e-6, T, '-')
    #plt.plot(t / 1e-6, T, '-')
    plt.ylabel('$T$ [K]')
    plt.xlabel('$t$ [us]')
    plt.subplot(nplots, 1, 5)
    plt.plot(t / 1e-6, p_ext / 1e5, '-')
    #plt.plot(t / 1e-6, T, '-')
    plt.ylabel('$p_{\mathrm{ext}}$ [bar]')
    plt.xlabel('$t$ [us]')
    plt.subplot(nplots, 1, 6)
    plt.plot(t / 1e-6, p_b / 1e5, '-')
    #plt.plot(t / 1e-6, T, '-')
    plt.ylabel('$p_{\mathrm{b}}$ [bar]')
    plt.xlabel('$t$ [us]')
    plt.subplot(nplots, 1, 7)
    plt.plot(t / 1e-6, R_equ / 1e-6, '-')
    #plt.plot(t / 1e-6, R_dot, '-')
    plt.ylabel('$R_{\mathrm{n}}$ [um]')
    
    plt.tight_layout()
    #plt.tight_layout(pad=0.2, w_pad=1.0, h_pad=0.5)
    plt.show()
    #plt.savefig('Toegel_20160314_Rmax500um_T20C.pdf')

    print("")
    
def test_KellerMiksis_ode():
    print("Keller-Miksis")
    print("=============")
    t, R, R_dot = bm.KellerMiksis_ode(10e-6, 0., 10e-6,
                                      0., 50., 0.0001)

    t = t[0:-1]
    R = R[0:-1]
    R_dot = R_dot[0:-1]
    
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title("Keller--Miksis")
    plt.plot(t / 1e-6, R / 1e-6, '-')
    plt.ylabel('$R$ [um]')
    plt.subplot(2, 1, 2)
    plt.plot(t / 1e-6, R_dot, '-')
    plt.ylabel('$\dot R$ [m/s]')
    plt.xlabel('$t$ [us]')
    plt.tight_layout()
    plt.show()

    print("")
    
if __name__ == '__main__':
    fig_width = 8.27   # width in inches
    fig_height = 11.69 # height in inches
    fig_size =  [fig_width, fig_height]
    params = {'figure.figsize': fig_size}
    plt.rcParams.update(params)

    #test_Gilmore_ode()
    #test_GilmoreEick_ode()
    #test_GilmoreEick()
    test_Toegel_ode()
    #test_KellerMiksis_ode()
    #    unittest.main()
