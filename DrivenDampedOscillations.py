#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:17:19 2021

@author: Lindsay LeBlanc

An interactive plot to examine the position, velocity, and acceleration of a 
simple harmonic oscillator.  Also, include the representation of these three 
quantities in the complex plane
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

# Set up the figure
fig = plt.figure(2,(12,9))
plt.clf()
fig, axs = plt.subplots(nrows = 2, ncols = 2,num=2,gridspec_kw={'width_ratios': [3, 2]})
plt.subplots_adjust(bottom=0.3,top = 0.95)


# Initial conditions for the oscillator
t = np.arange(0.0, 20.0, 0.001)
a0 = 1
amp = a0
f0 = 1
om0= 1
om_init = 1
gamma_init = 0.01
delta_f = 0.01
t0 = 0

def xposition(t,gamma,om0):
    if (gamma > om0):
        return np.exp(-gamma*t)*(0.5*np.exp(gamma*t*np.sqrt(1-(om0/gamma)**2))+0.5*np.exp(-gamma*t*np.sqrt(1-(om0/gamma)**2)))
    elif (gamma < om0):
        omNew = np.sqrt(om0**2 - gamma**2)
        return np.exp(-gamma*t)*np.cos(omNew*t)
    elif (gamma == om0):
        return np.exp(-om0*t)
    
def drivenResponse(t,gamma,om,om0):
    phi = np.arctan2(-gamma,(om0-om))
    return 1/om/np.sqrt((om-om0)**2 + gamma**2)*np.cos(om*t+phi)

def amplitudeResponse(om,gamma, om0):
    return 1/om/np.sqrt((om-om0)**2 + gamma**2)

def phaseResponse(om,gamma,om0):
    return np.arctan2(-gamma,(om0-om))

# Calculation with initial parameters
drive = 50*np.cos(om_init*t)
drive0 = 50*np.cos(om_init*t0)
position = a0 * drivenResponse(t,gamma_init,om_init,om0)
position0 = a0 * drivenResponse(t0,gamma_init,om_init,om0)

# Initial plots, set plot parameters
# position
centre, = axs[0,0].plot(t/2/np.pi,np.zeros_like(t), 'k-', alpha = 0.2)
dri, = axs[0,0].plot(t/2/np.pi, drive, 'crimson', lw=2)
driX, = axs[0,0].plot(t0/2/np.pi, drive0, 'ko', ms=5)
pos, = axs[0,0].plot(t/2/np.pi, position, 'dodgerblue', lw=2)
posX, = axs[0,0].plot(t0/2/np.pi, position0, 'ko', ms =5,)
axs[0,0].set_ylabel(r'position amplitude (arb. units)')
axs[0,0].set_xlabel(r'time ($2\pi/\omega_0$)')
axs[0,0].set_ylim([-110,110])


# mass-spring
force, = axs[1,0].plot([100,50*np.cos(om_init*t0)+100],[0.1,0.1],'r-',lw = 5,marker = 'o',ms = 10)
spring, = axs[1,0].plot([-100,drivenResponse(t0,gamma_init,om_init,om0)+100],[0,0],'k--',lw = 2)
mass, = axs[1,0].plot(drivenResponse(t0,gamma_init,om_init,om0)+100,0,'ks',ms = 15,mfc = 'dodgerblue')
forcenode, = axs[1,0].plot(100,0.1,'k',lw = 5,marker = 'o',ms = 10,mfc = 'w')
rect = Rectangle((-100,-0.3), 20, 0.6,facecolor="black", alpha=1)
axs[1,0].add_patch(rect)
axs[1,0].spines['right'].set_visible(False)
axs[1,0].spines['top'].set_visible(False)
axs[1,0].spines['left'].set_visible(False)
axs[1,0].set_yticks([])
axs[1,0].set_xticks([0,50,100,150,200])
axs[1,0].set_xticklabels(['-100','','0','','+100'])
axs[1,0].set_xlabel(r'position, (arb. units)')
axs[1,0].set_xlim([-110,240])


# Amplitude and phase (plot vs. freq)
oms = np.linspace(0.1,2,200)
# Amplitude
ampl, = axs[0,1].plot(oms, amplitudeResponse(oms,gamma_init,om0), 'm-')
ampl0, = axs[0,1].plot(om0, amplitudeResponse(om0,gamma_init,om0), 'ko', ms = 5)
axs[0,1].set_xlim([0.1,2])
axs[0,1].set_xlabel(r'angular frequency, $\omega$ ($\omega_0$)')
axs[0,1].set_ylabel('position amplitude (arb. units)')


# Amplitude
phase, = axs[1,1].plot(oms, phaseResponse(oms,gamma_init,om0)/np.pi, 'm-')
phase0, = axs[1,1].plot(om0, phaseResponse(om0,gamma_init,om0)/np.pi, 'ko', ms = 5)
axs[1,1].set_xlim([0.1,2])
axs[1,1].set_ylabel(r'position phase ($\pi$)')
axs[1,1].set_xlabel(r'angular frequency, $\omega$ ($\omega_0$)')


# Set up slider bars
axcolor = 'lightgoldenrodyellow'
axcolor2 = 'lightblue'
axomega = plt.axes([0.15, 0.15, 0.55, 0.03], facecolor=axcolor)
# axamp = plt.axes([0.15, 0.15, 0.55, 0.03], facecolor=axcolor)
axdamping = plt.axes([0.15, 0.1, 0.55, 0.03], facecolor=axcolor)
axtime = plt.axes([0.15, 0.05, 0.55, 0.03], facecolor=axcolor2)

som = Slider(axomega, r'Drive freq $\omega/\omega_0$', 0.5, 2, valinit=om_init, valstep=delta_f)
# samp = Slider(axamp, r'Amp ($A$)', 0.1, 1, valinit=a0)
stime = Slider(axtime, r'Time ($2\pi/\omega_0$)', 0, 20, valinit=t0)
sdamp = Slider(axdamping, r'Damping ($\gamma/\omega_0$)', 0, 0.1, valinit=gamma_init)


# Everything to update when the sliders change
def update(val):
    # amp = samp.val
    om = som.val
    time = stime.val
    damp = sdamp.val
    
    pos.set_ydata(amp*drivenResponse(t,damp,om,om0))
    posX.set_xdata(time/2/np.pi)
    posX.set_ydata(amp*drivenResponse(time,damp,om,om0))
    
    driX.set_xdata(time/2/np.pi)
    driX.set_ydata(50*np.cos(time*om))
    dri.set_ydata(50*np.cos(t*om))
    
    force.set_xdata([[100,50*np.cos(om_init*time)+100]])
    
    ampl.set_ydata(amplitudeResponse(oms,damp,om0))
    ampl0.set_ydata(amplitudeResponse(om,damp,om0))
    ampl0.set_xdata(om)
    
    phase.set_ydata(phaseResponse(oms,damp,om0)/np.pi)
    phase0.set_ydata(phaseResponse(om,damp,om0)/np.pi)
    phase0.set_xdata(om)
    
    mass.set_xdata(amp*drivenResponse(time,damp,om,om0)+100)
     
    spring.set_xdata([-100,amp*drivenResponse(time,damp,om,om0)+100])
    
    fig.canvas.draw_idle()
    
   
# Code to update sliders
som.on_changed(update)
# samp.on_changed(update)
sdamp.on_changed(update)
stime.on_changed(update)

resetax = plt.axes([0.8, 0.065, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

# Reset to initial values
def reset(event):
    sdamp.reset()
    stime.reset()
button.on_clicked(reset)


# Code for animation
playax = plt.axes([0.8, 0.125, 0.1, 0.04])
Playbutton = Button(playax, 'Play', color=axcolor, hovercolor='0.975')

def update_plot(frame):
    # Update for animation
    time = (stime.val)+1/10
    stime.set_val(time)


def Play(event):
    global ani
    stime.reset()
    ani = animation.FuncAnimation(fig, update_plot, frames = range(1, 200),interval=50,repeat = False)
    plt.show()
Playbutton.on_clicked(Play)


plt.show()

# # A second, static, separate plot with three cases
# # Set up the figure
# fig = plt.figure(3,(6,4))
# plt.clf()
# centreStatic, = plt.plot(t/2/np.pi,np.zeros_like(t), 'k-', alpha = 0.2)
# posStaticOver2, = plt.plot(t/2/np.pi, xposition(t,2,1), 'forestgreen', lw=3,label = r'$\gamma = 1.5\omega_0$ (over)')
# posStaticOver, = plt.plot(t/2/np.pi, xposition(t,1.5,1), 'darkgreen', lw=3,label = r'$\gamma = 2\omega_0$ (over)')
# posStaticCrit, = plt.plot(t/2/np.pi, xposition(t,1,1), 'black', lw=3,label = r'$\gamma = \omega_0$ (critical)')
# posStaticUnder, = plt.plot(t/2/np.pi, xposition(t,0.8,1), 'darkblue', lw=3,label = r'$\gamma = 0.8\omega_0$ (under)')
# posStaticUnder2, = plt.plot(t/2/np.pi, xposition(t,0.5,1), 'dodgerblue', lw=3,label = r'$\gamma = 0.5\omega_0$ (under)')
# posStaticLow, = plt.plot(t/2/np.pi, xposition(t,0.2,1), 'lightblue', lw=3,label = r'$\gamma = 0.2\omega_0$ (under)')
# plt.legend()
# plt.xlabel(r'time ($2\pi/\omega_0$)')
# plt.ylabel(r'amplitude ($x_0$)')

plt.savefig('DampedOscillator.png', dpi = 300)
