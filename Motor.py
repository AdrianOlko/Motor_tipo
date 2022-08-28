# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 17:12:38 2022

@author: Usuario
"""
import numpy as np
from pyturb.gas_models import isa
from pyturb.gas_models import SemiperfectIdealGas
from pyturb.gas_models import IsentropicFlow

sp_air = SemiperfectIdealGas('Air') 
is_flow = IsentropicFlow(sp_air)
Rg = sp_air.Rg

MC = 12.01070 #Masa molecular - carbono
MH = 1.00794  #Masa molecular - hidrógeno
MO = 15.99940 #Masa molecular - oxígeno
MN = 14.00670 #Masa molecular - nitrógeno
MAr = 39.94800 #Masa molecular - argón

Mg = MN*1.56168 + MO*0.419590 + MAr*0.009365 + MC*0.000319 # Estequiometria del aire

LHVA1 = 43.15e6 #J/kg JETA1
LHVFT = 44.2e6 #J/kg Fischer-Tropsch Synfuel
LHVMETANOL = 19.93e6 #J/kg Metano
LHVETANOL = 26.7e6 #J/kg Etano
LHVMETANO = 50e6 #J/kg Metano
LHVETANO = 47.794e6 #J/kg Etano
LHVH2 = 119.93e6 #J/kg Hidrogeno

A1 = [.5, 2.5, 10, 21, 19, 16, 12.5, 9.5, 6, 2.5, .5] #Numero de hidrocarburos(iso+par) de 7 a 17
BIO50 = [2, 7.5, 15.5, 21.5, 18, 12, 8.5, 6, 7, 1.5, .5] #https://pubs.acs.org/doi/10.1021/acsomega.1c04002?fig=fig2&ref=pdf
    

def cp_air(T):
    f = [1.009950160E+04, -1.968275610E+02, 5.009155110E+00, -5.761013730E-03, 1.066859930E-05,
        -7.940297970E-09, 2.185231910E-12] # 200 to 1000 K
    l = [2.415214430E+05, -1.257874600E+03, 5.144558670E+00, -2.138541790E-04, 7.065227840E-08,
        -1.071483490E-11, 6.577800150E-16] # 1000 to 6000 K
    if T <= 1000:
        return Rg*((f[0]*T**-2)+(f[1]*T**-1)+(f[2])+(f[3]*T)+(f[4]*T**2)+(f[5]*T**3)+(f[6]*T**4))
    if T > 1000:
        return Rg*((l[0]*T**-2)+(l[1]*T**-1)+(l[2])+(l[3]*T)+(l[4]*T**2)+(l[5]*T**3)+(l[6]*T**4))

def γ(T):
    return cp_air(T)/(cp_air(T)-Rg)

def c(h):
    T = isa.temperature_isa(h)
    return np.sqrt(γ(T)*Rg*T)

def stag_t(M, T):
    return T * (1 + (γ(T) - 1)/2 * M**2)

def stag_p(p, M, T):
    return p * (stag_t(M, T)/T) ** (γ(T)/(γ(T)-1))

def difusor1_2(𝜂_d, T_1, M, p_1):
    T_2t = stag_t(M, T_1)
    p_2t = p_1 * ((𝜂_d * (T_2t/T_1 - 1))  + 1)**(γ(T_1)/(γ(T_1) - 1))
    return T_2t, p_2t

def fan2_13(𝜂_fan, rc_fan, p_2t, T_2t):
    c = 0
    tol = 0.001
    T_13t = T_2t
    while c <= 100:
        c = c + 1
        tmp_0 = T_13t
        Tfan = (0.75*T_13t + 0.25*T_2t) # aproximacion 25 % entrada 75% salida
        T_13t = T_2t*((rc_fan**((γ(Tfan)-1)/γ(Tfan)) - 1)/𝜂_fan + 1)
                
        if np.abs((T_13t-tmp_0)/tmp_0)<tol:
            return T_13t, p_2t*rc_fan
        
def LPC13_25(𝜂_LPC, rc_LPC, p_13t, T_13t):
    c = 0
    tol = 0.001
    T_25t = T_13t
    while c <= 100:
        c += 1
        tmp_0 = T_25t
        Tcomp = (0.75*T_25t + 0.25*T_13t) # aproximacion 25 % entrada 75% salida
        T_25t = T_13t*((rc_LPC**((γ(Tcomp)-1)/γ(Tcomp)) - 1)/𝜂_LPC + 1)
                
        if np.abs((T_25t-tmp_0)/tmp_0)<tol:
            return T_25t, p_13t*rc_LPC

def HPC25_3(𝜂_HPC, rc_HPC, p_25t, T_25t):
    c = 0
    tol = 0.001
    T_3t = T_25t
    while c <= 100:
        c += 1
        tmp_0 = T_3t
        Tcomp = (0.75*T_3t + 0.25*T_25t) # aproximacion 25 % entrada 75% salida
        T_3t = T_25t*((rc_HPC**((γ(Tcomp)-1)/γ(Tcomp)) - 1)/𝜂_HPC + 1)
                
        if np.abs((T_3t-tmp_0)/tmp_0)<tol:
            return T_3t, p_25t*rc_HPC

def HPT4_45(𝜂_HPT, p_4t, T_4t, G_4, W_HPC, H_4t):
    c = 0
    tol = 0.001
    T_45t = T_4t
    while c <= 100:
        c += 1
        tmp_0 = T_45t
        Tturb = (0.4*T_45t + 0.6*T_4t) # aproximacion
        T_45t = (H_4t - W_HPC*1.005)/G_4/cp_air(Tturb) # rend mec .995
                
        H_45t = G_4*cp_air(Tturb)*T_45t
        W_HPT = H_45t - H_4t
                
        if np.abs((T_45t - tmp_0)/tmp_0)<tol:
            p_45t = p_4t * ((T_45t/T_4t - 1)/𝜂_HPT + 1)**(γ(Tturb)/(γ(Tturb)-1))
            return T_45t, p_45t, W_HPT, H_45t
        
def LPT45_5(𝜂_LPT, p_45t, T_45t, G_45, W_LPC, W_fan, H_45t):
    c = 0
    tol = 0.001
    T_5t = T_45t
    while c <= 100:
        c += 1
        tmp_0 = T_5t
        Tturb = (0.4*T_5t + 0.6*T_45t) # aproximacion
        T_5t = (H_45t - (W_LPC + W_fan)*1.005)/G_45/cp_air(Tturb) # rend mec .995
        
        H_5t = G_45*cp_air(Tturb)*T_5t
        W_LPT = H_5t - H_45t
                
        if np.abs((T_5t - tmp_0)/tmp_0)<tol:
            p_5t = p_45t * ((T_5t/T_45t - 1)/𝜂_LPT + 1)**(γ(Tturb)/(γ(Tturb)-1))
            return T_5t, p_5t, W_LPT, H_5t

def tobera17_18(𝜂_tobs, T_17t, p_17t, p_0, adapt=1):
    p0 = adapt*p_0
    T_18 = T_17t * ((𝜂_tobs * (((p0/p_17t)**((γ(T_17t)-1)/γ(T_17t)))-1))+1)                
    v_18 = np.sqrt(2*cp_air(T_17t)*(T_17t - T_18))
    return T_18, v_18  

def tobera5_8(𝜂_tobp, T_5t, p_5t, p_0, adapt=1):
    p0 = adapt*p_0
    T_8 = T_5t * ((𝜂_tobp * (((p0/p_5t)**((γ(T_5t)-1)/γ(T_5t)))-1))+1)
    
    if T_5t<=T_8:
        v_8 = 0
    else:
        v_8 = np.sqrt(2*cp_air(T_5t)*(T_5t - T_8))
    return T_8, v_8   

def fest(Comb):
    
    def est(percent):    
        fest = 0
        α = 7
        β = 2*α + 2
        for i in percent:
            fest += (i/100)*((α*MC + β*MH)/(4.76*Mg*(α + β/4)))
            α += 1
            β = 2*α + 2
        return fest
    
    def hidroc(C, O):
        α = C
        β = 2*α + 2        
        fest = (α*MC + β*MH + O*MO)/(4.76*Mg*(α + β/4 - O/2))
        return fest
    
    if Comb == 0:
        f_est = est(A1)
        L = LHVA1
        return f_est, L
    
    if Comb == 1:
        f_est = est(BIO50)
        L = 0.5*LHVA1 + 0.5*LHVFT
        return f_est, L
    
    if Comb == 2:
        f_est = hidroc(1,1)
        L = LHVMETANOL
        return f_est, L
    
    if Comb == 3:
        f_est = hidroc(2,1)
        L = LHVETANOL
        return f_est, L
    
    if Comb == 4:
        f_est = hidroc(1,0)
        L = LHVMETANO
        return f_est, L
    
    if Comb == 5:
        f_est = hidroc(2,0)
        L = LHVETANO
        return f_est, L
    
    if Comb == 6:
        f_est = (4*MH)/(4.76*Mg)
        L = LHVH2
        return f_est, L

def CC3_4(𝜂_CC, rc_CC, p_3t, T_3t, G_3, MAXT=2000, fuel=0):
    c = 0
    tol = 0.001
    var = 0.2 # % del dosado estequiometrico
    f_est, L = fest(fuel)
    fcc = var * f_est
    T_4t = (fcc*L*𝜂_CC + cp_air(T_3t)*T_3t) / (cp_air(T_3t)*(1+fcc)) #valor de partida
    while c <= 100:
        c += 1
        fcc_0 = var * f_est # % del dosado estequiometrico
        tmp_0 = T_4t
        T_4t = (fcc_0*L*𝜂_CC + cp_air(T_3t)*T_3t) / (cp_air(tmp_0)*(1+fcc_0))
                 
        if ((np.abs(T_4t - tmp_0))/tmp_0)<tol:
            if T_4t<=MAXT:
                var += 0.0001
                c = 0
            else:
                return T_4t, p_3t*rc_CC, G_3 + (G_3*fcc_0), G_3*fcc_0, L       

def Nivel_tech(N):
    Rendimientos = [[1,1,1,1,1,1,1,1],
                    [.9,.78,.8,.9,.88,.8,.8,.95],
                    [.95,.82,.84,.92,.94,.83,.85,.97],
                    [.98,.86,.88,.94,.99,.87,.89,.98],
                    [.995,.89,.9,.95,.99,.89,.9,.995]]
                    #Difusor,fan,compresores,rc_cc,comb,turbalta,turbaja,Toberas
                    # añadir tobera y difusor
    return Rendimientos[N]

def Motor_tipo(Comb, Λ, M, alt, tempc, rend):
    R_1 = 1 #m
    A_1 = np.pi*R_1**2 # m^2
    M_0 = M #M
    Rend = Nivel_tech(rend)
    Fuel = Comb #[0-4]
    Tcc = tempc

    h = alt
    v_0 = M_0 * c(h)
    p_0 = isa.pressure_isa(h)
    T_0 = isa.temperature_isa(h)
    G_0 = isa.density_state_eq(h)*A_1*v_0

    v_1 = v_0
    p_1 = p_0
    T_1 = T_0
    G_1 = G_0

    T_2t, p_2t = difusor1_2(Rend[0], T_1, M_0, p_1)

    rc_fan = 1.45 #<-----
    T_13t, p_13t = fan2_13(Rend[1], rc_fan, p_2t, T_2t)
    G_13 = G_2 = G_1
    H_13t = G_13 * cp_air(T_13t) * T_13t
    W_fan = (H_13t - (G_13 * cp_air(T_2t) * T_2t))
    
    G_Primario = G_13/Λ
    G_Secundario = G_13 - G_Primario

    rc_LPC = 2 #<-----
    T_25t, p_25t = LPC13_25(Rend[2], rc_LPC, p_13t, T_13t)
    G_25 = G_Primario
    H_25t = G_25 * cp_air(T_25t) * T_25t
    W_LPC = H_25t - (G_25 * cp_air(T_13t) * T_13t)

    rc_HPC = 10 #<-----
    T_3t, p_3t = HPC25_3(Rend[2], rc_HPC, p_25t, T_25t)
    G_3 = .97*G_25 #<----- sang
    H_3t = G_3 * cp_air(T_3t) * T_3t
    W_HPC = H_3t - (G_3 * cp_air(T_25t) * T_25t)

    T_4t, p_4t, G_4, c_cc, L = CC3_4(Rend[4], Rend[3], p_3t, T_3t, G_3, Tcc, Fuel)
    H_4t = G_4 * cp_air(T_4t) * T_4t

    G_4r = G_4 + 0.03*G_25 #<----- sang

    T_45t, p_45t, W_HPT, H_45t = HPT4_45(Rend[5], p_4t, T_4t, G_4r, W_HPC, H_4t)

    G_45 = G_4r

    T_5t, p_5t, W_LPT, H_5t = LPT45_5(Rend[6], p_45t, T_45t, G_45, W_LPC, W_fan, H_45t)

    T_17t = T_13t
    p_17t = p_13t
    T_18, v_18 = tobera17_18(Rend[7], T_17t, p_17t, p_0, 1.05)

    T_8, v_8 = tobera5_8(Rend[7], T_5t, p_5t, p_0, 1.05)

    ##Actuaciones

    E_18 = (G_Secundario*v_18 - G_Secundario*v_0)
    E_8 = (G_45*v_8 - G_Primario*v_0)
    E_total = E_18 + E_8
    E_sp = E_total/G_Primario
    Ce = c_cc/E_total
    Isp = (1/Ce)/9.80665
    𝜂_m = (((G_Secundario*v_18**2)+(G_45*v_8**2)) - G_0*v_0**2)/(2*c_cc*L)
    𝜂_p = (2*v_0*(((G_Secundario*v_18)+(G_45*v_8))-G_0*v_0))/(((G_Secundario*v_18**2)+(G_45*v_8**2))-G_0*v_0**2)
    𝜂_mp = 𝜂_m * 𝜂_p

    return E_total, E_sp, Ce*10**6, Isp, 𝜂_m, 𝜂_p, 𝜂_mp

    #rint('E={0:6.0f} N, Ie= {1:4.1f} m/s, Ce= {2:1.8f} kg/s/N, rend_motor= {3:1.3f}, rend_prop= {4:0.3f}, rend_mp= {5:1.3f}'
     #     .format(E_total, Ie, Ce, 𝜂_m, 𝜂_p, 𝜂_mp))