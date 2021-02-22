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
fig = plt.figure(2,(8,7))
plt.clf()
fig, axs = plt.subplots(nrows = 2, ncols = 1,num=2)
plt.subplots_adjust(bottom=0.3,top = 0.95)


# Initial conditions for the oscillator
t = np.arange(0.0, 20.0, 0.001)
a0 = 1
amp = a0
f0 = 1
om0= 1
om = om0
gamma = 1
delta_f = 0.1
t0 = 0

def xposition(t,gamma,om0):
    if (gamma > om0):
        return np.exp(-gamma*t)*(0.5*np.exp(gamma*t*np.sqrt(1-(om0/gamma)**2))+0.5*np.exp(-gamma*t*np.sqrt(1-(om0/gamma)**2)))
    elif (gamma < om0):
        omNew = np.sqrt(om0**2 - gamma**2)
        return np.exp(-gamma*t)*np.cos(omNew*t)
    elif (gamma == om0):
        return np.exp(-om0*t)

# Calculation with initial parameters
position = a0 * xposition(t,gamma,om0)
position0 = a0 * xposition(t0,gamma,om0)

# Initial plots, set plot parameters
# position
centre, = axs[0].plot(t/2/np.pi,np.zeros_like(t), 'k-', alpha = 0.2)
pos, = axs[0].plot(t/2/np.pi, position, 'dodgerblue', lw=2)
posX, = axs[0].plot(t0/2/np.pi, position0, 'ko', ms =5,)
axs[0].set_ylabel(r'position ($A$)')
axs[0].set_xlabel(r'time ($2\pi/\omega_0$)')
axs[0].set_ylim([-1.1,1.1])

# mass-spring
spring, = axs[1].plot([-1,xposition(t0,gamma,om0)+1],[0,0],'k--',lw = 2)
mass, = axs[1].plot(xposition(t0,gamma,om0)+1,0,'ks',ms = 15,mfc = 'dodgerblue')
rect = Rectangle((-1,-0.3), 0.2, 0.6,facecolor="black", alpha=1)
axs[1].add_patch(rect)
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].spines['left'].set_visible(False)
axs[1].set_yticks([])
axs[1].set_xticks([0,0.5,1,1.5,2])
axs[1].set_xticklabels(['-1','','0','','+1'])
axs[1].set_xlabel(r'position, $x$ ($A$)')




# Set up slider bars
axcolor = 'lightgoldenrodyellow'
axcolor2 = 'lightblue'
# axomega = plt.axes([0.15, 0.15, 0.55, 0.03], facecolor=axcolor)
# axamp = plt.axes([0.15, 0.15, 0.55, 0.03], facecolor=axcolor)
axdamping = plt.axes([0.15, 0.1, 0.55, 0.03], facecolor=axcolor)
axtime = plt.axes([0.15, 0.05, 0.55, 0.03], facecolor=axcolor2)

# som = Slider(axomega, r'Angular freq $\omega_0$', 0.5, 2, valinit=om0, valstep=delta_f)
# samp = Slider(axamp, r'Amp ($A$)', 0.1, 1, valinit=a0)
stime = Slider(axtime, r'Time ($2\pi/\omega_0$)', 0, 20, valinit=t0)
sdamp = Slider(axdamping, r'Damping ($\gamma/\omega_0$)', 0, 3, valinit=gamma)


# Everything to update when the sliders change
def update(val):
    # amp = samp.val
    # om = som.val
    time = stime.val
    damp = sdamp.val
    
    pos.set_ydata(amp*xposition(t,damp,om))
    posX.set_xdata(time)
    posX.set_ydata(amp*xposition(time,damp,om))
    
    mass.set_xdata(amp*xposition(time,damp,om)+1)
     
    spring.set_xdata([-1,amp*xposition(time,damp,om)+1])
    
    fig.canvas.draw_idle()
    
   
# Code to update sliders
# som.on_changed(update)
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

# A second, static, separate plot with three cases
# Set up the figure
fig = plt.figure(3,(6,4))
plt.clf()
centreStatic, = plt.plot(t/2/np.pi,np.zeros_like(t), 'k-', alpha = 0.2)
posStaticOver2, = plt.plot(t/2/np.pi, xposition(t,2,1), 'forestgreen', lw=3,label = r'$\gamma = 1.5\omega_0$ (over)')
posStaticOver, = plt.plot(t/2/np.pi, xposition(t,1.5,1), 'darkgreen', lw=3,label = r'$\gamma = 2\omega_0$ (over)')
posStaticCrit, = plt.plot(t/2/np.pi, xposition(t,1,1), 'black', lw=3,label = r'$\gamma = \omega_0$ (critical)')
posStaticUnder, = plt.plot(t/2/np.pi, xposition(t,0.8,1), 'darkblue', lw=3,label = r'$\gamma = 0.8\omega_0$ (under)')
posStaticUnder2, = plt.plot(t/2/np.pi, xposition(t,0.5,1), 'dodgerblue', lw=3,label = r'$\gamma = 0.5\omega_0$ (under)')
posStaticLow, = plt.plot(t/2/np.pi, xposition(t,0.2,1), 'lightblue', lw=3,label = r'$\gamma = 0.2\omega_0$ (under)')
plt.legend()
plt.xlabel(r'time ($2\pi/\omega_0$)')
plt.ylabel(r'amplitude ($x_0$)')

plt.savefig('DampedOscillator.png', dpi = 300)
