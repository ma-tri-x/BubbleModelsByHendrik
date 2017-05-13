# Copyright 2017 Hendrik Soehnholz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Bubble Models

Solve ODEs of various bubble models
using scipy.integrate.odeint()
or scipy.integrate.ode

Hendrik Soehnholz
Initial version: 2014-09-01
"""

import numpy as np
#import matplotlib.pyplot as plt
#import pylab as pl
from scipy.integrate import odeint, ode
from scipy.optimize import brentq
import logging

#
# initial conditions and parameters
#

t0 = 1e-6 # Skalierung der Zeit [s]
#R0 = 1.036e-3 # Anfangsradius [m]
#v0 = 0. # Anfangsgeschw. [m/s]
#Requ = 100e-6 # Ruheradius [m]
pac = 0e5 # Schalldruck [Pa]
frequ = 0e3 # Frequenz [Hz]
omega = 2 * np.pi * frequ # Kreisfrequenz [rad]
T_l = 293.15 # Wassertemperatur [Kelvin] fuer Modell von Toegel et al.

k_B = 1.38066e-23 # Boltzmann-Konstante [J / K]
avogadro = 6.02214e23 # Avogadro-Konstante [Teilchenzahl / mol]
R_g = 8.31451 # universelle Gaskonstante [J / (mol * K)]

pstat = 1e5 # statischer Druck [Pa]
rho0 = 998.21 # Dichte der Fluessigkeit [kg / m ^ 3]
mu = 1002e-6 # Viskositaet der Fluessigkeit [Pa * s]
#mu = 653.2e-6 # bei T = 40 degC
#mu = 353.8e-6 # bei T = 80 degC
sigma = 0.07275 # Oberflaechenspannung [N / m]
#sigma = 0.06267 # bei T = 80 degC
#Btait = 3213e5 # B aus Tait-Gleichung. Woher kommt dieser Wert fuer B?
Btait = 3046e5 # B aus Tait-Gleichung
ntait = 7.15 # n aus Tait-Gleichung
c0 = 1483. # Schallgeschwindigkeit [m / s]
#c0 = np.sqrt((pstat + Btait) * ntait) # Schallgeschw. [m / s]

# Polytropenexponent
kappa = 4./3. # adiabatisch
#kappa = 1. # isotherm

# van-der-Waals Konstanten
# b (fuer Gilmore, Eick, Keller--Miksis)
#bvan = 0.0016 # welches Gas?
bvan = 0.0015 # Luft, aus Koch et al., Comp. Fluids 126, 71-90 (2016)
#bvan = 0.
# a (fuer Toegel Modell)
#a_van = 8.86 # aus Toegel Paper. Welches Gas?
#a_van = 8.75 # Luft
#a_van = 9.23 # Wasser
a_van = 11.63 # Wasserstoff

# Dampf in der Blase
#pvapour = 7073. # Dampfdruck [Pa]
T0_Kelvin = 273.15 # Ice point [K]
theta_v = [2295, 5255, 5400] # Vibrations-Freiheitsgrade des Wassermolekuels

# Gas in der Blase
mu_gas = 0.0000171 # Viskositaet von Luft [Pa * s]
c_p = 1005. # spez. Waerme bei konstantem Druck fuer Luft [J / (kg * K)]
lambda_g = 0.0262 # Waermeleitfaehigkeit von Luft [W / (m * K)]
#T_gas_0 = 273. + T_l # Gastemperatur am Anfang [K]
#print T_gas_0
lambda_mix = 0.0248 # Waermeleitfaehigkeit (thermal conductivity)
                    # von Wasserdampf [W / (m * K)]
#lambda_mix = 0.001 # Test

#n_R = 1e12 # Teilchenzahldichte im Gleichgewicht
#D = 2.8e-5 # Diffusionskonstante [m ^ 2 / s]
D = 3.57e-5
#D = 0.
chi = 1.8e-5 # Temperaturleitfaehigkeit (thermal diffusivity) [m ^ 2 / s]

# Skalierung
#scale_t = T0 # Zeit
#scale_R = Requ # Blasenradius
#scale_U = Requ / T0 # Geschwindigkeit
#scale_p = scale_U * scale_U * rho0 # Druck

# Parameter skalieren
# sc_pstat = pstat / scale_p
# sc_pac = pac / scale_p
# sc_pvapour = pvapour / scale_p
# sc_sigma = sigma / scale_R / scale_p
# sc_mu = mu / scale_R / scale_p * scale_U
# sc_Btait = Btait / scale_p
# sc_frequ = frequ * scale_t
# sc_omega = 2. * np.pi * sc_frequ
# sc_pequ = sc_pstat + 2. * sc_sigma - sc_pvapour
# sc_c0 = np.sqrt((sc_pstat + sc_Btait) * ntait)


# sc_mu_gas = mu_gas / scale_R / scale_p * scale_U
# sc_c_p = c_p / scale_U
# sc_lambda_g = lambda_g / rho0 \
#     / scale_U / scale_U / scale_U / scale_U / scale_t
# sc_Re = np.sqrt(sc_pstat - sc_pvapour) / sc_mu_gas
# sc_Pr = sc_mu_gas * sc_c_p / sc_lambda_g
# sc_Nu = 0.111 * np.sqrt(sc_Re) * sc_Pr ** (1. / 3.)


# Anfangswerte skalieren
# Anfangsgasdruck in der Blase [Pa]
#p0 = (pstat + 2. * sigma / Requ - pvapour) \
#    * ((1. - bvan) / ((R0 / Requ) ** 3. - bvan)) ** kappa

#R0 = R0 / scale_R; # Anfangsradius (skaliert)
#v0 = v0 / scale_U # Anfangsgeschwindigkeit (skaliert)
#p0 = p0 / scale_p # Anfangsgasdruck in der Blase (skaliert)
#p0 = sc_pequ

#print p0

def create_tdata(t_start, t_end, t_step):
    """generate scaled time data
    """

#    return np.linspace(t_start / scale_t, t_end / scale_t, \
#                           (t_end - t_start) / t_step)
    return np.arange(t_start / scale_t, t_end / scale_t, t_step / scale_t)

def set_scale(t0, Requ):
    """Determine scaling factors."""
    global scale_t, scale_R, scale_U, scale_p

    scale_t = t0 # time
    scale_R = Requ # bubble radius
    scale_U = scale_R / scale_t # bubble wall velocity
    scale_p = scale_U * scale_U * rho0 # pressure

    return

def scale_parameters(pvapour_in):
    """Scale parameters according to scaling factors."""

    global sc_pstat, sc_pac, sc_pvapour, sc_Btait
    global sc_sigma, sc_mu
    global sc_pequ
    global sc_frequ, sc_omega
    global sc_c0
    global sc_mu_gas, sc_c_p, sc_lambda_g, sc_Re, sc_Pr, sc_Nu

    sc_pstat = pstat / scale_p
    sc_pac = pac / scale_p
    sc_pvapour = pvapour_in / scale_p
    sc_Btait = Btait / scale_p

    sc_sigma = sigma / scale_R / scale_p
    #sc_mu = mu / scale_R / scale_p * scale_U
    sc_mu = mu / (scale_p * scale_t)

    sc_pequ = sc_pstat + 2. * sc_sigma - sc_pvapour

    sc_frequ = frequ * scale_t
    sc_omega = 2. * np.pi * sc_frequ

    sc_c0 = np.sqrt((sc_pstat + sc_Btait) * ntait)

    sc_mu_gas = mu_gas / scale_R / scale_p * scale_U
    sc_c_p = c_p / scale_U
    sc_lambda_g = lambda_g / rho0 \
        / scale_U / scale_U / scale_U / scale_U / scale_t
    sc_Re = np.sqrt(sc_pstat - sc_pvapour) / sc_mu_gas
    sc_Pr = sc_mu_gas * sc_c_p / sc_lambda_g
    sc_Nu = 0.111 * np.sqrt(sc_Re) * sc_Pr ** (1. / 3.)

    return

def scale_initconds(R0_in, v0_in, Requ, pvapour):
    """Scale initial conditions according to scaling factors."""

    global R0, v0, p0

    R0 = R0_in / scale_R
    v0 = v0_in / scale_U

    # Achtung:
    # p0 wird aus unskalierten Werten berechnet!
    p0 = (pstat + 2. * sigma / Requ - pvapour) \
      * ((1. - bvan) / ((R0_in / Requ) ** 3. - bvan)) ** kappa
    #print("p_g(0) = {0} Pa".format(p0))
    p0 = p0 / scale_p

    return

def get_vapour_pressure(T):
    """Vapour pressure of water as a function of the temperature

    Equation from
    W. Wagner und A. Prusz, J. Phys. Chem. Ref. Data 31, 387--535 (2002)
    Section 2.3.1

    Temperature scale: ITS-90
    """

    # Parameters
    pc = 22.064e6 # [Pa]
    Tc = 647.096 # [K]
    a1 = -7.85951783
    a2 = 1.84408259
    a3 = -11.7866497
    a4 = 22.6807411
    a5 = -15.9618719
    a6 = 1.80122502

    # Conversion degree Celsius -> Kelvin
    #T0_Kelvin = 273.15 # [K]
    T = T + T0_Kelvin

    theta = 1 - T / Tc

    # Compute vapour pressure pv
    # as a function of the temperature T
    pv = pc * np.exp(Tc / T * (a1 * theta \
                                   + a2 * theta ** 1.5 \
                                   + a3 * theta ** 3 \
                                   + a4 * theta ** 3.5 \
                                   + a5 * theta ** 4 \
                                   + a6 * theta ** 7.5))

    return pv

def GilmoreEick_deriv(x, t):
    """Compute one integration step
    using the extended Gilmore equations
    with additional equation for the gas pressure inside the bubble.
    """

    global T

    R = x[0]
    R_dot = x[1]
    pg = x[2]

    pinf = sc_pstat - sc_pac * np.sin(sc_omega * t)
    pinf_dot = -sc_pac * sc_omega * np.cos(sc_omega * t)

    T_gas = T_gas_0 * pg * R ** 3 / sc_pequ
    # if (t < 1.):
    #     print pg
    #     print T_gas
    T = np.append(T, [t, T_gas])
    pb = pg + sc_pvapour # Druck in der Blase
    pg_dot = - 3. * kappa * pg * R * R * R_dot \
        / (R ** 3 - bvan) \
        + 1.5 * (kappa - 1.) * sc_lambda_g * sc_Nu \
        * (T_gas_0 - T_gas) / R / R

    p = pb - (2.* sc_sigma + 4. * sc_mu * R_dot) / R

    p_over_pinf = (p + sc_Btait) / (pinf + sc_Btait)

    H = ntait / (ntait - 1.) * (pinf + sc_Btait) \
        * (p_over_pinf ** (1. - 1. / ntait) - 1.)
    H1 = p_over_pinf ** (- 1. / ntait)
    H2 = p_over_pinf ** (1. - 1. / ntait) / (ntait - 1.) \
        - ntait / (ntait - 1.)
    C = np.sqrt(sc_c0 * sc_c0 + (ntait - 1.) * H)

    dR = R_dot
    dR_dot = (- 0.5 * (3. - R_dot / C) * R_dot * R_dot \
                    + (1. + R_dot / C) * H \
                    + (1. - R_dot / C) * R \
                    * (H1 * (pg_dot \
                                 + (2. * sc_sigma + 4. * sc_mu * R_dot) \
                                 * R_dot / R / R) \
                           + H2 * pinf_dot) / C) \
                           / ((1. - R_dot / C) \
                                  * (R + 4. * sc_mu \
                                         * p_over_pinf ** (-1. / ntait) / C))
    dpg = pg_dot
    return [dR, dR_dot, dpg]

def Gilmore_deriv(x, t):
    """Compute one integration step
    using the Gilmore equations.
    """

    global p_gas

    R = x[0]
    R_dot = x[1]

    pinf = sc_pstat - sc_pac * np.sin(sc_omega * t)
    pinf_dot = -sc_pac * sc_omega * np.cos(sc_omega * t)

    pg = (sc_pstat + 2. * sc_sigma - sc_pvapour) \
    * ((1. - bvan) / (R ** 3. - bvan)) ** kappa
#    print pg
    p_gas = np.append(p_gas, [t, pg])
    pb = pg + sc_pvapour # Druck in der Blase
    pg_dot = - 3. * kappa * pg * R * R * R_dot / (R ** 3 - bvan)
    p = pb - (2.* sc_sigma + 4. * sc_mu * R_dot) / R

    p_over_pinf = (p + sc_Btait) / (pinf + sc_Btait)

    H = ntait / (ntait - 1.) * (pinf + sc_Btait) \
        * (p_over_pinf ** (1. - 1. / ntait) - 1.)
    H1 = p_over_pinf ** (- 1. / ntait)
    H2 = p_over_pinf ** (1. - 1. / ntait) / (ntait - 1.) \
        - ntait / (ntait - 1.)
    C = np.sqrt(sc_c0 * sc_c0 + (ntait - 1.) * H)

    dR = R_dot
    dR_dot = (- 0.5 * (3. - R_dot / C) * R_dot * R_dot \
                    + (1. + R_dot / C) * H \
                    + (1. - R_dot / C) * R \
                    * (H1 * (pg_dot \
                                 + (2. * sc_sigma + 4. * sc_mu * R_dot) \
                                 * R_dot / R / R) \
                           + H2 * pinf_dot) / C) \
                           / ((1. - R_dot / C) \
                                  * (R + 4. * sc_mu \
                                         * p_over_pinf ** (-1. / ntait) / C))

    return (dR, dR_dot)

def GilmoreEick(R0_in, v0_in, Requ, \
                    t_start, t_end, t_step, \
                    T_l=20.):
    """Run the calculation (Gilmore + Eick)
    with the given initial conditions and parameters.
    returns: t, R, R_dot, pg, T, i
    """

    global T
    global T_gas_0, sc_pvapour

    T_gas_0 = T0_Kelvin + T_l # initial gas temperature inside bubble [K]

    # Compute vapour pressure using liquid temperature T_l
    pvapour_in = get_vapour_pressure(T_l)
    print("p_v = {0} Pa".format(pvapour_in))

    # scale initial conditions and parameters
    set_scale(Requ)

    # parameters
    scale_parameters(pvapour_in)
    #print pvapour_in, sc_pvapour

    # initial conditions
    scale_initconds(R0_in, v0_in, Requ, pvapour_in)
    #print scale_R, R0

    # solve system of ODEs
    T = np.zeros(0)
    t_data = create_tdata(t_start, t_end, t_step)

    xsol = odeint(GilmoreEick_deriv, [R0, v0, p0], t_data,
                  #full_output = True,
                 )

    R = xsol[:, 0] * scale_R
    R_dot = xsol[:, 1] * scale_U
    pg = xsol[:, 2] * scale_p
    t = t_data * scale_t
    T = np.reshape(T, (-1, 2))

#    np.savetxt('GilmoreEick_result.dat', (t / 1e-6, R / 1e-6, R_dot, pg), \
#                   delimiter = '\t')
#    np.savetxt('GilmoreEick_Temp.dat', (T[:, 0], T[:, 1]))

    return (t, R, R_dot, pg, T)

#
# nochmal fuer das normale Gilmore-Modell (ohne Erweiterung)
#
def Gilmore(R0_in, v0_in, Requ, \
                t_start, t_end, t_step, \
                T_l=20.):
    """Run the calculation (Gilmore)
    with the given initial conditions and parameters.
    returns: t, R, R_dot, pg, i
    """

    global p_gas

    # Compute vapour pressure using liquid temperature T_l
    pvapour_in = get_vapour_pressure(T_l)
    print "pv = ", pvapour_in

    # scale initial conditions and parameters
    set_scale(Requ)

    # parameters
    scale_parameters(pvapour_in)
#    print pvapour_in, sc_pvapour

    # initial conditions
    scale_initconds(R0_in, v0_in, Requ, pvapour_in)
#    print scale_R, R0

    # solve system of ODEs
    p_gas = np.zeros(0)
    t_data = create_tdata(t_start, t_end, t_step)

#    print (R0, v0)

    xsol, i = odeint(Gilmore_deriv, (R0, v0), t_data, full_output=True)

    R = xsol[:, 0] * scale_R
    R_dot = xsol[:, 1] * scale_U
    p_gas = np.reshape(p_gas, (-1, 2))
    t = t_data * scale_t

#    np.savetxt('Gilmore_result.dat', (t / 1e-6, R / 1e-6, R_dot))
#    np.savetxt('Gilmore_pg.dat', (p_gas[:, 0], p_gas[:, 1]))

    return (t, R, R_dot, p_gas, i)


def Gilmore_equation(t, x):
    """Compute one integration step
    using the Gilmore equations.
    """

    global p_gas

    R = x[0]
    R_dot = x[1]

    pinf = sc_pstat - sc_pac * np.sin(sc_omega * t)
    pinf_dot = -sc_pac * sc_omega * np.cos(sc_omega * t)

    pg = (sc_pstat + 2. * sc_sigma - sc_pvapour) \
    * ((1. - bvan) / (R ** 3. - bvan)) ** kappa
    #print pg
    p_gas = np.append(p_gas, [t, pg])
    pb = pg + sc_pvapour # Druck in der Blase
    pg_dot = - 3. * kappa * pg * R * R * R_dot / (R ** 3 - bvan)
    p = pb - (2.* sc_sigma + 4. * sc_mu * R_dot) / R

    p_over_pinf = (p + sc_Btait) / (pinf + sc_Btait)

    H = ntait / (ntait - 1.) * (pinf + sc_Btait) \
        * (p_over_pinf ** (1. - 1. / ntait) - 1.)
    H1 = p_over_pinf ** (- 1. / ntait)
    H2 = p_over_pinf ** (1. - 1. / ntait) / (ntait - 1.) \
        - ntait / (ntait - 1.)
    C = np.sqrt(sc_c0 * sc_c0 + (ntait - 1.) * H)

    dR = R_dot
    dR_dot = (- 0.5 * (3. - R_dot / C) * R_dot * R_dot \
                    + (1. + R_dot / C) * H \
                    + (1. - R_dot / C) * R \
                    * (H1 * (pg_dot \
                                 + (2. * sc_sigma + 4. * sc_mu * R_dot) \
                                 * R_dot / R / R) \
                           + H2 * pinf_dot) / C) \
                           / ((1. - R_dot / C) \
                                  * (R + 4. * sc_mu \
                                         * p_over_pinf ** (-1. / ntait) / C))

    return [dR, dR_dot]

def Gilmore_ode(R0_in, v0_in, Requ, \
                t_start, t_end, t_step, \
                T_l=20.):
    """Solve Gilmore ODE in single steps using scipy.integrate.ode
    """

    global p_gas

    # Compute vapour pressure using liquid temperature T_l
    pvapour_in = get_vapour_pressure(T_l)
    print "pv = ", pvapour_in

    # scale initial conditions and parameters
    set_scale(t0, Requ)

    # parameters
    scale_parameters(pvapour_in)
#    print pvapour_in, sc_pvapour

    # initial conditions
    scale_initconds(R0_in, v0_in, Requ, pvapour_in)
#    print scale_R, R0

    # solve system of ODEs
    p_gas = np.zeros(0)
    t_data = create_tdata(t_start, t_end, t_step)

#    print (R0, v0)

    #xsol, i = odeint(Gilmore_deriv, (R0, v0), t_data, full_output = True)
    o = ode(Gilmore_equation).set_integrator('dopri5',
                                             #atol=[1e-6, 1e0],
                                             #rtol=[1e-3, 1e-3],
                                             #first_step=1e-9,
                                             #verbosity=1,
                                            )
    o.set_initial_value([R0, v0], t_start)

    nsteps = (t_end - t_start) / t_step + 1
    t = np.zeros(nsteps)
    R = np.zeros(nsteps)
    R_dot = np.zeros(nsteps)
    i = 0
    R_prev = R0
    growing = False
    while o.successful() and o.t < t_end:
        o.integrate(o.t + t_step)
        #print("%g\t%g\t%g" % (o.t, o.y[0], o.y[1]))
        t[i] = o.t * scale_t
        R[i] = o.y[0] * scale_R
        R_dot[i] = o.y[1] * scale_U
        i += 1

        if o.y[0] >= R_prev:
            growing = True
#            print('Bubble is growing...')
        elif o.y[0] < R_prev and growing:
            # max. reached
            #print('max!')

            # decrease Requ (condensation, diffusion)
            R0_in = o.y[0] * scale_R
            v0_in = o.y[1] * scale_U
            #Requ = 0.6 * Requ
            set_scale(t0, Requ)
            scale_parameters(pvapour_in)
            scale_initconds(R0_in, v0_in, Requ, pvapour_in)
            o.set_initial_value([R0, v0], o.t)

            growing = False
        R_prev = o.y[0]

    #plt.figure()
    #plt.axis([0, 100, 0, 600])
    #plt.plot(t / 1e-6, R / 1e-6, '.')
    #plt.show()

#    R = xsol[:, 0] * scale_R
#    R_dot = xsol[:, 1] * scale_U
#    p_gas = np.reshape(p_gas, (-1, 2))
#    t = t_data * scale_t

    return t, R, R_dot


def GilmoreEick_equation(t, x):
    """Compute one integration step
    using the Gilmore--Eick equations.
    """

    global T

    R = x[0]
    R_dot = x[1]
    pg = x[2]

    pinf = sc_pstat - sc_pac * np.sin(sc_omega * t)
    pinf_dot = -sc_pac * sc_omega * np.cos(sc_omega * t)

    T_gas = T_gas_0 * pg * R ** 3 / sc_pequ
    # if (t < 1.):
    #     print pg
    #     print T_gas
    T = np.append(T, [t, T_gas])
    pb = pg + sc_pvapour # Druck in der Blase
    pg_dot = - 3. * kappa * pg * R * R * R_dot \
        / (R ** 3 - bvan) \
        + 1.5 * (kappa - 1.) * sc_lambda_g * sc_Nu \
        * (T_gas_0 - T_gas) / R / R

    p = pb - (2.* sc_sigma + 4. * sc_mu * R_dot) / R

    p_over_pinf = (p + sc_Btait) / (pinf + sc_Btait)

    H = ntait / (ntait - 1.) * (pinf + sc_Btait) \
        * (p_over_pinf ** (1. - 1. / ntait) - 1.)
    H1 = p_over_pinf ** (- 1. / ntait)
    H2 = p_over_pinf ** (1. - 1. / ntait) / (ntait - 1.) \
        - ntait / (ntait - 1.)
    C = np.sqrt(sc_c0 * sc_c0 + (ntait - 1.) * H)

    dR = R_dot
    dR_dot = (- 0.5 * (3. - R_dot / C) * R_dot * R_dot \
                    + (1. + R_dot / C) * H \
                    + (1. - R_dot / C) * R \
                    * (H1 * (pg_dot \
                                 + (2. * sc_sigma + 4. * sc_mu * R_dot) \
                                 * R_dot / R / R) \
                           + H2 * pinf_dot) / C) \
                           / ((1. - R_dot / C) \
                                  * (R + 4. * sc_mu \
                                         * p_over_pinf ** (-1. / ntait) / C))
    dpg = pg_dot
    return [dR, dR_dot, dpg]

def GilmoreEick_ode(R0_in, v0_in, Requ, \
                    t_start, t_end, t_step, \
                    T_l=20.):
    """Solve Gilmore--Eick ODE in single steps using scipy.integrate.ode.
    Decrease Requ by a factor during each rebound.
    """

    global T
    global T_gas_0, sc_pvapour

    # initial gas temperature inside bubble [K]
    T_gas_0 = T0_Kelvin + T_l

    # Compute vapour pressure using liquid temperature T_l
    pvapour_in = get_vapour_pressure(T_l)
    print "pv = ", pvapour_in

    # scale initial conditions and parameters
    set_scale(t0, Requ)

    # parameters
    scale_parameters(pvapour_in)

    # initial conditions
    scale_initconds(R0_in, v0_in, Requ, pvapour_in)

    # solve system of ODEs
    T = np.zeros(0)
#    t_data = create_tdata(t_start, t_end, t_step)

    o = ode(GilmoreEick_equation).set_integrator('dopri5',
                                                 #atol=[1e-6, 1e0],
                                                 #rtol=[1e-3, 1e-3],
                                                 #first_step=1e-9,
                                                 #verbosity=1,
                                                )
    o.set_initial_value([R0, v0, p0], t_start)

    nsteps = (t_end - t_start) / t_step + 1
    t = np.zeros(nsteps)
    R = np.zeros(nsteps)
    R_dot = np.zeros(nsteps)
    pg = np.zeros(nsteps)
    i = 0
    R_prev = R0
    growing = False
    while o.successful() and o.t < t_end:
        o.integrate(o.t + t_step)
#        print("%g\t%g\t%g\t%g" % (o.t, o.y[0], o.y[1], o.y[2]))
        t[i] = o.t * scale_t
        R[i] = o.y[0] * scale_R
        R_dot[i] = o.y[1] * scale_U
        pg[i] = o.y[2] * scale_p
        i += 1

        if o.y[0] >= R_prev:
            growing = True
#            print('Bubble is growing...')
        elif o.y[0] < R_prev and growing:
            # max. reached
            #print('Max. radius in rebound reached!')

            # decrease Requ (condensation, diffusion)
            R0_in = o.y[0] * scale_R
            v0_in = o.y[1] * scale_U
            #Requ = 0.66 * Requ
            set_scale(t0, Requ)
            scale_parameters(pvapour_in)
            scale_initconds(R0_in, v0_in, Requ, pvapour_in)
            o.set_initial_value([R0, v0, p0], o.t)

            growing = False
        R_prev = o.y[0]

#    plt.figure()
#    plt.axis([0, 100, 0, 600])
#    plt.plot(t / 1e-6, R / 1e-6, '.')
#    plt.show()

    T = np.reshape(T, (-1, 2))

    return t, R, R_dot, pg, T

def Toegel_equation(t, x):
    """Compute one integration step
    using the equations from Toegel et al., Phys. Rev. Lett. 85, 3165 (2000).
    """

    #global p_g_prev # letzter Wert fuer Druck in der Blase
    #global T_l # Wassertemperatur [Kelvin]
    #global dt # Zeitschritt [s]
    global sc_p_inf, sc_p_g, sc_R_equ

    sc_R = x[0]
    sc_R_dot = x[1]
    sc_N = x[2]
    sc_T = x[3]
    #print("R = {0:g}".format(R))
    #print("R_dot = {0:g}".format(R_dot))
    #print("N = {0:g}".format(N))
    #print("T = {0:g}".format(T))

    sc_S = 4 * np.pi * sc_R ** 2 # Oberflaeche der Blase
    sc_V = 4 * np.pi * sc_R ** 3 / 3 # Volumen der Blase

    # Zeitschritt [scale_t * s]
    #delta_t = np.min([R / np.abs(R_dot), 1e-6 / scale_t])


    # Teilchenzahl darf nicht negativ sein.
    if sc_N < 0:
        logging.warning("N < 0. Setze N = 0.")
        sc_N = 0


    def f(sc_R_equ):
        """Zusammenhang zwischen Ruheradius R0 und Teilchenzahl N
        """

        return sc_pstat * (1. - 1. / a_van ** 3) * sc_R_equ ** 3 \
          + 2 * sc_sigma * (1. - 1. / a_van ** 3) * sc_R_equ ** 2 \
          - 3 * (sc_N + sc_N_Ar) * sc_R_g * scale_N * sc_T_l / (4 * np.pi)

    #print(f(0.1e-6 / scale_R))
    #print(f(1e-3 / scale_R))

    # Eine Nullstelle von f(R_equ) finden
    # (Intervall muss angegeben werden!)
    sc_R_equ = brentq(f, 0.1e-6 / scale_R, 1e-3 / scale_R)
    #R_equ = 10e-6 / scale_R
    #print("R_equ = {0:g}".format(R_equ))

    # Radius muss groesser als Hardcore-Radius sein
    if sc_R < sc_R_equ / a_van:
        logging.warning("R < R_equ / a_van.")
        #R = R_equ

    # Druck in der Blase
    # Beim ersten Schritt ist p_g_prev noch nicht gesetzt.
    #if 'p_g_prev' not in globals():
    #if np.abs(p_g_prev) <= 2e-6:
    #    print("Reset pgprev to equilibrium pressure.")
    #    p_g_prev = sc_pstat + 2 * sc_sigma / R_equ # - pvapour

    # Oben wird schon geprueft, ob N >= 0 ist.
    sc_p_g = (sc_N + sc_N_Ar) * sc_R_g * scale_N * sc_T \
             / ((sc_R ** 3 - (sc_R_equ / a_van) ** 3) * 4 * np.pi / 3)

    #print("p_g = {0:g} = {1:g} Pa".format(p_g, p_g * scale_p))

    #p_g_dot_alt = (p_g - p_g_prev)
    #print("p_g_dot_alt = {0:g}".format(p_g_dot_alt))
    #p_g_prev = p_g

    # externe Anregung
    sc_p_inf = -sc_pac * np.sin(sc_omega * t)
    sc_p_inf_dot = -sc_pac * sc_omega * np.cos(sc_omega * t)
    #print("sc_p_inf = {0}".format(sc_p_inf))

    #
    # Teilchenzahl
    #

    # Diffusionskoeffizient bestimmen
    # Werte aus Bird, Stewart, Lightfoot: Transport Phenomena (2002)
    # und woher noch?

    # Molmassen
    M_vapour = 18.01628 # [g / mol], Wasserdampf
    #M_gas = 39.948 # [g / mol], Argon
    #M_gas = 31.999 # [g / mol], Sauerstoff (O2)
    M_gas = 2.016 # [g / mol], Wasserstoff (H2)
    #M_gas = 28.964 # [g / mol], Luft

    # Lennard--Jones Parameter
    sigma_vapour = 2.824 # [Angstroem], Wasserdampf
    #sigma_gas = 3.432 # [Angstroem], Argon
    #sigma_gas = 3.433 # [Angstroem], Sauerstoff (O2)
    sigma_gas = 2.915 # [Angstroem], Wasserstoff (H2)
    #sigma_gas = 3.617 # [Angstroem], Luft
    sigma_AB = (sigma_vapour + sigma_gas) / 2. # [Angstroem]
    epsK_vapour = 230.9 # [K], Wasserdampf
    #epsK_gas = 122.4 # [K], Argon
    #epsK_gas = 113.0 # [K], Sauerstoff (O2)
    epsK_gas = 38.0 # [K], Wasserstoff (H2)
    #epsK_gas = 97.0 # [K], Luft
    epsK_AB = np.sqrt(epsK_vapour * epsK_gas) # [K]

    # Kollisionsintegral
    #Omega_AB = 1. # haengt von epsK und T_l ab!
    # Fit aus Transport Phenomena (2002)
    def omega_diff(kT_eps):
        return 1.06036 / kT_eps ** 0.15610 \
               + 0.19300 / np.exp(0.47635 * kT_eps) \
               + 1.03587 / np.exp(1.52996 * kT_eps) \
               + 1.76474 / np.exp(3.89411 * kT_eps)
    Omega_AB = omega_diff(T_l / epsK_AB)
    #print("Omega = {0:g}".format(Omega_AB))

    # N in mol angeben!
    # n in mol / cm^3 angeben!
    D = 2.2646e-5 * np.sqrt(T_l * (1 / M_vapour + 1 / M_gas)) \
        / (sigma_AB ** 2 * Omega_AB \
        * (sc_n_R + sc_N_Ar / sc_V) * scale_N / scale_R ** 3 * 1e-6) \
        * 1e-4 # [m^2 / s]
    #D = D * 8. # Test
    #D = 0
    sc_D = D * scale_t / scale_R ** 2
    #print("D = {0:g}\tsc_D = {1:g}".format(D, sc_D))

    #l_diff = np.max([np.min([np.sqrt(sc_D * R / np.abs(R_dot)), R / np.pi]),
    #                        50e-9 / scale_R])
    sc_l_diff = np.min([np.sqrt(sc_D * sc_R / np.abs(sc_R_dot)), sc_R / np.pi])
    #l_diff = 1.
    #print("l_diff = {0:g} m".format(l_diff * scale_R))
    sc_n = sc_N / sc_V
    dN = sc_S * sc_D * (sc_n_R - sc_n) / sc_l_diff
    #print("dN = {0:g}".format(dN))

    #print("delta_t = {0:g}".format(delta_t))
    # Minimalwert N0 aus dem Dampfdruck bestimmen
    p_v = get_vapour_pressure(T_l - T0_Kelvin)
    N0 = (p_v * sc_V * scale_R ** 3) \
          / (R_g * T_l)
    sc_N0 = N0 / scale_N
    #print("N0 = {0:g}".format(N0 * avogadro))
    #print("sc_N0 = {0:g}".format(sc_N0))

    #if (sc_N + dN < sc_N0):
        #pass
        #logging.warning("N + dN < N0")
        #print("N = {0:g}\tdN = {1:g}".format(N, dN))
        #print("N0 * dt = {0:g}".format(sc_N0 * delta_t))
        #dN = -N
        #dN = 0.
        #dN = -0.5 * (N - sc_N0)
        #dN = N * delta_t
    # test
    #dN = 0.

    #
    # Temperatur
    #

    # chi aus n_R, N_Ar / V und lambda_mix berechnen
    chi = lambda_mix / ((4 * sc_n_R + 2.5 * sc_N_Ar / sc_V) * R_g \
          * scale_N / scale_R ** 3)
    #chi = 0
    sc_chi = chi * scale_t / scale_R ** 2
    #print("lambda = {0:g}\tchi = {1:g}".format(lambda_mix, chi))
    #print("rho_cp = {0:g}".format((4 * sc_n_R + 2.5 * sc_N_Ar / sc_V) * R_g \
    #      * scale_N / scale_R ** 3))

    sc_l_th = np.min([np.sqrt(sc_chi * sc_R / np.abs(sc_R_dot)), sc_R / np.pi])
    #l_th = 1
    #sc_lambda_mix = 0
    sc_Q_dot = sc_S * sc_lambda_mix * (sc_T_l - sc_T) / sc_l_th
    sc_V_dot = sc_S * sc_R_dot

    # Druck in der Blase
    #p_b = -(p_g + sc_pvapour \
    #      - sc_p_inf - sc_pstat \
    #      - 4. * sc_mu * R_dot / R - 2. * sc_sigma / R)
    #p_b = p_g + sc_pvapour
    sc_p_b = sc_p_g
    #p_b = 0

    sumT1 = 0
    sumT2 = 0
    for th in theta_v:
        sc_th = th / scale_T
        sumT1 += (sc_th / sc_T) / (np.exp(sc_th / sc_T) - 1)
        sumT2 += (sc_th / sc_T) ** 2 * np.exp(sc_th / sc_T) \
                 / (np.exp(sc_th / sc_T) - 1) ** 2

    #C_v = 1.5 * N_Ar * R_g * scale_N + (3. + sumT2) * N * R_g * scale_N
    sc_C_v = (1.5 * sc_N_Ar + (3. + sumT2) * sc_N) * sc_R_g * scale_N
    dT = (sc_Q_dot - sc_p_b * sc_V_dot \
          + (4. * sc_T_l - 3. * sc_T - sc_T * sumT1) \
          * dN * sc_R_g * scale_N) / sc_C_v

    #if np.abs(dT) > 10:
    #    print("dT = {0:g}".format(dT))

    # Ableitung Druck in der Blase
    sc_p_g_dot = sc_R_g * scale_N \
                 / (4 * np.pi / 3 * (sc_R ** 3 - (sc_R_equ / a_van) ** 3)) \
                 * (dN * sc_T + sc_N * dT + sc_N_Ar * dT) \
                 - sc_R_g * scale_N / (4 * np.pi / 3) * (sc_N + sc_N_Ar) * sc_T \
                 * (3 * sc_R ** 2 * sc_R_dot) \
                 / ((sc_R ** 3 - (sc_R_equ / a_van) ** 3) ** 2)

    # Wenn R_dot und sc_c0 gleich sind, wird durch Null geteilt!
    if sc_R_dot == sc_c0:
        logging.warning("R_dot == c0")
        #R_dot = R_dot * 1.01

    dR = sc_R_dot
    if sc_R + dR < 0:
        pass
        #logging.warning("R + dR < 0")
        #dR = -0.5 * (R - R_equ / a_van)
    dR_dot = (-1.5 * sc_R_dot ** 2 * (1. - sc_R_dot / (3. * sc_c0)) \
              + (1. + sc_R_dot / sc_c0) * (sc_p_g - sc_p_inf - sc_pstat) \
              - 4. * sc_mu * sc_R_dot / sc_R \
              - 2. * sc_sigma / sc_R \
              + sc_R * (sc_p_g_dot + sc_p_inf_dot) / sc_c0) \
              / ((1. - sc_R_dot / sc_c0) * sc_R + 4 * sc_mu / sc_c0)

    #print("dR = {0}".format(dR))
    #print("dR_dot = {0}".format(dR_dot))
    #print("dN = {0}".format(dN))
    #print("dT = {0}".format(dT))

    return [dR, dR_dot, dN, dT]

def Toegel_ode(R0, v0, T0, R_equ_Ar, t_start, t_end, t_step):
    """Toegel Modell
    Initialisierung und Aufruf des Loesers
    (jeder Zeitschritt einzeln)
    """
    #global p_g_prev
    global T_l
    global sc_n_R, sc_N_Ar
    #global R_equ_Ar

    #global dt
    global scale_t, scale_R, scale_U, scale_p, scale_N, scale_T
    global sc_c0
    global sc_frequ, sc_omega, sc_pac
    global sc_pstat
    global sc_mu, sc_sigma
    global sc_D, sc_chi, sc_lambda_mix
    global sc_k_B, sc_R_g
    global sc_pvapour
    global sc_T_l

    # T0 in K umrechnen
    T0 = T0_Kelvin + T0
    T_l = T0

    # Anfangswert fuer N berechnen
    p = get_vapour_pressure(T0 - T0_Kelvin)
    #p = get_vapour_pressure(21.0)
    #p = pstat
    N0 = (p * 4 * np.pi * R0 ** 3 / 3) \
          / (R_g * T0)
    print("N0 = {0:g}".format(N0 * avogadro))

    # Skalierung
    # time scale
    if frequ != 0:
        scale_t = 1 / frequ
    else:
        scale_t = 1e-6
    scale_R = R0 # bubble radius
    scale_U = scale_R / scale_t # bubble wall velocity
    scale_p = scale_U * scale_U * rho0 # pressure
    #scale_N = 1e-14
    scale_N = N0
    scale_T = T0_Kelvin
    print("scale_t = {0}".format(scale_t))
    print("scale_R = {0}".format(scale_R))
    print("scale_U = {0}".format(scale_U))
    print("scale_p = {0}".format(scale_p))
    print("scale_N = {0}".format(scale_N))
    print("scale_T = {0}".format(scale_T))

    sc_t_start = t_start / scale_t
    sc_t_end = t_end / scale_t
    sc_t_step = t_step / scale_t
    sc_R0 = R0 / scale_R
    sc_v0 = v0 / scale_U
    sc_N0 = N0 / scale_N
    sc_T0 = T0 / scale_T

    sc_T_l = T_l / scale_T
    sc_R_equ_Ar = R_equ_Ar / scale_R

    sc_c0 = c0 / scale_U
    sc_frequ = frequ * scale_t
    sc_omega = 2 * np.pi * sc_frequ
    sc_pac = pac / scale_p
    sc_pstat = pstat / scale_p
    pvapour = get_vapour_pressure(T_l - T0_Kelvin)
    sc_pvapour = pvapour / scale_p

    print("pac = {0} bar".format(pac / 1e5))
    print("sc_pac = {0}".format(sc_pac))

    #p_g_prev = 1e-6

    #mu = 0
    #sigma = 0
    sc_mu = mu / (scale_p * scale_t)
    sc_sigma = sigma / (scale_R * scale_p)

    sc_R_g = R_g * scale_T / (scale_p * scale_R ** 3)
    sc_k_B = k_B * scale_T / (scale_p * scale_R ** 3)

    # Teilchenzahldichte im Gleichgewicht [1 / m^3]
    sc_n_R = sc_pvapour / (sc_R_g * scale_N * sc_T_l)
    #n_R = pvapour / (R_g * T_l) * scale_R ** 3 / scale_N
    sc_N_Ar = (sc_pstat + 2 * sc_sigma / sc_R_equ_Ar) \
              * (4 * np.pi * sc_R_equ_Ar ** 3 / 3) \
              / (sc_R_g * scale_N * sc_T_l)
    print("n_R = {0:g}".format(sc_n_R * scale_N * avogadro))
    print("N_Ar = {0:g}".format(sc_N_Ar * scale_N * avogadro))

    sc_D = D * scale_t / scale_R ** 2
    sc_chi = chi * scale_t / scale_R ** 2
    #print(sc_chi)
    sc_lambda_mix = lambda_mix * scale_T * scale_t / (scale_p * scale_R ** 2)

    #dt = t_step * scale_t
    #dt = t_step

    o = ode(Toegel_equation).set_integrator('dopri5',
                                            #method='bdf',
                                            #with_jacobian=True,
                                            #ixpr=True,
                                            #atol=[1e-6, 1e-6, 1e-6, 1e-6],
                                            #rtol=[1e-6, 1e-6, 1e-6, 1e-6],
                                            #atol=1e-3,
                                            #rtol=1e-2,
                                            #first_step=1e-6,
                                            #min_step=1e-18,
                                            #max_step=1e-5,
                                            #ifactor=3.0,
                                            #dfactor=3.0,
                                            #beta=0.1,
                                            #nsteps=3000,
                                            #order=15,
                                            verbosity=1,
                                           )
    o.set_initial_value([sc_R0, sc_v0, sc_N0, sc_T0], sc_t_start)

    nsteps = (sc_t_end - sc_t_start) / sc_t_step + 1
    t = np.zeros(nsteps)
    R = np.zeros(nsteps)
    R_dot = np.zeros(nsteps)
    N = np.zeros(nsteps)
    T = np.zeros(nsteps)
    p_ext = np.zeros(nsteps)
    p_b = np.zeros(nsteps)
    R_equ = np.zeros(nsteps)
    i = 0
    #R_prev = R0
    #growing = False
    while o.successful() and o.t < sc_t_end:
        o.integrate(o.t + sc_t_step)
        #print("%g\t%g\t%g\t%g\t%g" % (o.t, o.y[0], o.y[1], o.y[2], o.y[3]))
        t[i] = o.t
        R[i] = o.y[0]
        R_dot[i] = o.y[1]
        N[i] = o.y[2]
        T[i] = o.y[3]
        p_ext[i] = sc_pstat + sc_p_inf
        p_b[i] = sc_p_g
        R_equ[i] = sc_R_equ
        i += 1

    t = t * scale_t
    R = R * scale_R
    R_dot = R_dot * scale_U
    N = N * scale_N
    T = T * scale_T
    p_ext = p_ext * scale_p
    p_b = p_b * scale_p
    R_equ = R_equ * scale_R

    #print(R.dtype)

    return t, R, R_dot, N, T, p_ext, p_b, R_equ

def KellerMiksis_equation(t, x):
    """Compute one integration step
    using the equations from Keller and Miksis.
    """

    R = x[0]
    R_dot = x[1]

    #print("R = {0}".format(R))
    #print("R_dot = {0}".format(R_dot))

    #R_equ = 10e-6 / scale_R

    # Druck in der Blase
    p_g = (sc_pstat - sc_pvapour + 2 * sc_sigma / R_equ) \
          * (R_equ / R) ** (3 * kappa)
    print("p_g = {0:g} = {1:g} Pa".format(p_g, p_g * scale_p))
    p_g_dot = -3 * kappa * p_g * R_dot / R

    # angelegtes Schallfeld
    sc_p_inf = - sc_pac * np.sin(sc_omega * t + 0 * np.pi)
    sc_p_inf_dot = - sc_pac * sc_omega * np.cos(sc_omega * t + 0 * np.pi)

    dR = R_dot
    dR_dot = (-1.5 * R_dot ** 2 * (1. - R_dot / (3. * sc_c0)) \
              + (1. + R_dot / sc_c0) \
                * (p_g + sc_pvapour - sc_p_inf - sc_pstat) \
              - 4. * sc_mu * R_dot / R \
              - 2. * sc_sigma / R \
              + R * (p_g_dot - sc_p_inf_dot) / sc_c0) \
              / ((1. - R_dot / sc_c0) * R + 4 * sc_mu / sc_c0)

    #print("dR = {0}".format(dR))
    #print("dR_dot = {0}".format(dR_dot))

    return [dR, dR_dot]


def KellerMiksis_ode(R0, v0, R_equ_in, t_start, t_end, t_step):
    #global dt
    global R_equ
    global sc_c0
    global sc_frequ, sc_omega, sc_pac
    global sc_pstat, sc_pvapour
    global sc_mu, sc_sigma
    global sc_D, sc_chi, sc_lambda_mix
    global sc_k_B

    pvapour = get_vapour_pressure(20.)

    # Skalierung
    set_scale(1e-6, 1e-6) # t in 1e-6 s, R in 1e-6 m

    sc_R0 = R0 / scale_R
    sc_v0 = v0 / scale_U
    R_equ = R_equ_in / scale_R

    sc_c0 = c0 / scale_U
    sc_frequ = frequ * scale_t
    sc_omega = 2 * np.pi * sc_frequ
    sc_pac = pac / scale_p
    sc_pstat = pstat / scale_p
    sc_pvapour = pvapour / scale_p
    sc_mu = mu / (scale_p * scale_t)
    sc_sigma = sigma / (scale_R * scale_p)

    #dt = t_step * scale_t
    #dt = t_step

    o = ode(KellerMiksis_equation).set_integrator('dopri5',
                                                  #atol=[1e-6, 1e-1, 1e10, 1e-1],
                                                  #rtol=[1e-3, 1e-3],
                                                  #first_step=1e-9,
                                                  #nsteps=1000,
                                                  verbosity=1,
                                                 )
    o.set_initial_value([sc_R0, sc_v0], t_start)

    nsteps = (t_end - t_start) / t_step + 1
    t = np.zeros(nsteps)
    R = np.zeros(nsteps)
    R_dot = np.zeros(nsteps)
    i = 0
    #R_prev = R0
    #growing = False
    while o.successful() and o.t < t_end:
        o.integrate(o.t + t_step)
#        print("%g\t%g\t%g\t%g" % (o.t, o.y[0], o.y[1], o.y[2]))
        t[i] = o.t
        R[i] = o.y[0]
        R_dot[i] = o.y[1]
        i += 1

    t = t * scale_t
    R = R * scale_R
    R_dot = R_dot * scale_U

    return t, R, R_dot
