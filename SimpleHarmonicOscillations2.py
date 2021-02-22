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
fig = plt.figure(1,(8,10))
plt.clf()
fig, axs = plt.subplots(nrows = 3, ncols = 2,num=1)
plt.subplots_adjust(bottom=0.3,top = 0.95)


# Initial conditions for the oscillator
t = np.arange(0.0, 3.0, 0.001)
a0 = 1
f0 = 1
phi0= 0
delta_f = 0.1
t0 = 0


# Calculation with initial parameters
position = a0 * np.cos(2 * np.pi * f0 * t+ phi0*np.pi)
position0 = a0 * np.cos(2 * np.pi * f0 * t0+ phi0*np.pi)

velocity = -a0 * np.sin(2 * np.pi * f0 * t+ phi0*np.pi)
velocity0 = -a0 * np.sin(2 * np.pi * f0 * t0+ phi0*np.pi)

acceleration = -a0 * np.cos(2 * np.pi * f0 * t+ phi0*np.pi)
acceleration0 = -a0 * np.cos(2 * np.pi * f0 * t0+ phi0*np.pi)

# Initial plots, set plot parameters
# position
pos, = axs[0,0].plot(t, position, 'dodgerblue', lw=2)
posX, = axs[0,0].plot(t0, position0, 'ko', ms =5,)
axs[0,0].set_ylabel(r'position ($A$)')

# velocity
vel, = axs[1,0].plot(t, velocity, 'forestgreen',lw=2)
velX, = axs[1,0].plot(t0, velocity0, 'ko', ms =5)
axs[1,0].set_ylabel(r'velocity ($A\omega$)')

# acceleration
acc, = axs[2,0].plot(t, velocity, 'crimson',lw=2)
accX, = axs[2,0].plot(t0, velocity0, 'ko', ms =5)
axs[2,0].set_ylabel(r'acceleration ($A\omega^2$)')
axs[2,0].set_xlabel(r'time ($2\pi/\omega$)')

# mass-spring
spring, = axs[0,1].plot([-1,position0+1],[0,0],'k--',lw = 2)
mass, = axs[0,1].plot(position0+1,0,'ks',ms = 15,mfc = 'dodgerblue')
rect = Rectangle((-1,-0.3), 0.2, 0.6,facecolor="black", alpha=1)
axs[0,1].add_patch(rect)
axs[0,1].spines['right'].set_visible(False)
axs[0,1].spines['top'].set_visible(False)
axs[0,1].spines['left'].set_visible(False)
axs[0,1].set_yticks([])
axs[0,1].set_xticks([0,0.5,1,1.5,2])
axs[0,1].set_xticklabels(['-1','','0','','+1'])
axs[0,1].set_xlabel(r'position, $x$ ($A$)')

axs[1,1].axis('off')

# Complex plane
axs[2,1].axis('off')
ax = fig.add_subplot(224, projection='polar')
#
cxplaneX, = ax.plot([(phi0)*np.pi,(phi0)*np.pi],[0,1],color='dodgerblue',lw = 2,marker = 'o',ms = 5,mfc = 'w')
cxplaneV, = ax.plot([(phi0+0.5)*np.pi,(phi0+0.5)*np.pi],[0,1],color='forestgreen',lw = 2,marker = 'o',ms = 5,mfc = 'w')
cxplaneA, = ax.plot([(phi0+1)*np.pi,(phi0+1)*np.pi],[0,1],color='crimson',lw = 2,marker = 'o',ms = 5,mfc = 'w')
#
cxplaneprojX, = ax.plot([0+np.pi*(position0<0),0+np.pi*(position0<0)],[0,np.abs(position0)],'--',color = 'dodgerblue',lw = 2,marker = 'o',ms = 5,mec ='k')
cxplaneprojV, = ax.plot([0+np.pi*(velocity0<0),0+np.pi*(velocity0<0)],[0,np.abs(velocity0)],'--',color = 'forestgreen',lw = 2,marker = 'o',ms = 5,mec ='k')
cxplaneprojA, = ax.plot([0+np.pi*(acceleration0<0),0+np.pi*(acceleration0<0)],[0,np.abs(acceleration0)],'--',color = 'crimson',lw = 2,marker = 'o',ms = 5,mec ='k')
cxplaneprojK, = ax.plot(0,0,'ko',ms = 5,mec ='k')
#
plt.gca().set_thetagrids(angles = (0,90,180,270),labels=('+1',r'$+i$',r'$-1$',r'$-i$'))
plt.gca().set_rgrids((0.5,1),labels=('',''))
plt.gca().set_ylim([0,1.1])
axs[0,1].set_xlim([-1,3])
# ax.margins(x=0)

# Set up slider bars
axcolor = 'lightgoldenrodyellow'
axcolor2 = 'lightblue'
axfreq = plt.axes([0.15, 0.15, 0.55, 0.03], facecolor=axcolor)
axamp = plt.axes([0.15, 0.2, 0.55, 0.03], facecolor=axcolor)
axphase = plt.axes([0.15, 0.1, 0.55, 0.03], facecolor=axcolor)
axtime = plt.axes([0.15, 0.05, 0.55, 0.03], facecolor=axcolor2)

sfreq = Slider(axfreq, r'Freq, $f = \omega/2\pi$', 0.5, 2, valinit=f0, valstep=delta_f)
samp = Slider(axamp, r'Amp ($A$)', 0.1, 1, valinit=a0)
sphase = Slider(axphase, r'Phase $\phi$ ($\pi$)', 0, 2, valinit=phi0)
stime = Slider(axtime, r'Time ($2\pi/\omega$)', 0, 3, valinit=t0)


# Everything to update when the sliders change
def update(val):
    amp = samp.val
    freq = sfreq.val
    phi = sphase.val
    time = stime.val
    
    pos.set_ydata(amp*np.cos(2*np.pi*freq*t+phi*np.pi))
    posX.set_xdata(time)
    posX.set_ydata(amp*np.cos(2*np.pi*freq*time+phi*np.pi))
    
    vel.set_ydata(-amp*np.sin(2*np.pi*freq*t+phi*np.pi))
    velX.set_xdata(time)
    velX.set_ydata(-amp*np.sin(2*np.pi*freq*time+phi*np.pi))
    
    acc.set_ydata(-amp*np.cos(2*np.pi*freq*t+phi*np.pi))
    accX.set_xdata(time)
    accX.set_ydata(-amp*np.cos(2*np.pi*freq*time+phi*np.pi))
    
    mass.set_xdata(amp*np.cos(2*np.pi*freq*time+phi*np.pi)+1)
    
    cxplaneX.set_xdata(2*np.pi*freq*time+phi*np.pi)
    cxplaneV.set_xdata(2*np.pi*freq*time+phi*np.pi+1.57)
    cxplaneA.set_xdata(2*np.pi*freq*time+phi*np.pi+3.14)
    
    cxplaneX.set_ydata([0,amp])
    cxplaneV.set_ydata([0,amp])
    cxplaneA.set_ydata([0,amp])
    
    cxplaneprojX.set_xdata(0+np.pi*(np.cos(2*np.pi*freq*time+phi*np.pi)<0))
    cxplaneprojV.set_xdata(0+np.pi*(-np.sin(2*np.pi*freq*time+phi*np.pi)<0))
    cxplaneprojA.set_xdata(0+np.pi*(-np.cos(2*np.pi*freq*time+phi*np.pi)<0))
    
    cxplaneprojX.set_ydata([0,np.abs(amp*np.cos(2*np.pi*freq*time+phi*np.pi))])
    cxplaneprojV.set_ydata([0,np.abs(-amp*np.sin(2*np.pi*freq*time+phi*np.pi))])
    cxplaneprojA.set_ydata([0,np.abs(-amp*np.cos(2*np.pi*freq*time+phi*np.pi))])
    
    spring.set_xdata([-1,amp*np.cos(2*np.pi*freq*time+phi*np.pi)+1])
    
    fig.canvas.draw_idle()
    
   
# Code to update sliders
sfreq.on_changed(update)
samp.on_changed(update)
sphase.on_changed(update)
stime.on_changed(update)

resetax = plt.axes([0.8, 0.065, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

# Reset to initial values
def reset(event):
    sfreq.reset()
    samp.reset()
    sphase.reset()
    stime.reset()
button.on_clicked(reset)


# Code for animation
playax = plt.axes([0.8, 0.125, 0.1, 0.04])
Playbutton = Button(playax, 'Play', color=axcolor, hovercolor='0.975')

def update_plot(frame):
    # Update for animation
    time = (stime.val)+1/100
    stime.set_val(time)


def Play(event):
    global ani
    stime.reset()
    ani = animation.FuncAnimation(fig, update_plot, frames = range(1, 300),interval=50,repeat = False)
    plt.show()
Playbutton.on_clicked(Play)


plt.show()

