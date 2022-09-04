import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import pandas as pd

#import imageio

import streamlit as st


st.set_page_config(page_title='Arc sim', page_icon=':snowflake:', layout="wide", initial_sidebar_state="collapsed", menu_items=None)


def rangle(x,y,deg):
    return x*math.cos(math.radians(deg)) + y *math.sin(math.radians(deg))

def dist(x,y,deg):
    return abs(x*math.cos(math.radians(deg))-y*math.sin(math.radians(deg)))

def rot_point(x,y,deg,size):
    x = x - size//2
    y = y - size//2
    
    #print(rx,ry)
    rx = x*math.cos(math.radians(deg)) + y *math.sin(math.radians(deg))
    ry = y*math.cos(math.radians(deg)) - x *math.sin(math.radians(deg))
    #print(rx,ry)
    return rx+size//2,ry+size//2

def draw_bar(canv,size,deg,depth,thick,rad):
    
    cir_bord = math.sqrt(2)*rad
    
    for x in range(-int(cir_bord),int(cir_bord)): #assumes circ target
    #for x in range(-size//2,size//2): scans the whole vol
        for y in range(-size//2,size//2):
            if dist(x,y,deg) < rad:
                if rangle(y,x,deg) < depth and rangle(y,x,deg) > depth-thick:
                    #canv[y+size//2,x+size//2]+=1
                    canv[y+size//2,x+size//2] = np.nansum([canv[y+size//2,x+size//2],1])
    return canv
    
def arc_plot(canv,size,rad,teeth_locs=None):

    max_num = int(np.nanmax(canv))

    cmap = mpl.colors.ListedColormap(['white',
                                     '#1f77b4',
                                     '#d62728',
                                     '#e377c2',
                                     '#17becf',
                                     '#9467bd',
                                     '#ff7f0e',
                                     '#7f7f7f',
                                     '#2ca02c',
                                     '#bcbd22',
                                     '#1f77b4',
                                     '#d62728',
                                     '#e377c2',
                                     '#17becf',
                                     '#9467bd',
                                     '#ff7f0e',
                                     '#7f7f7f',
                                     '#2ca02c',
                                     '#bcbd22'][:max_num+1])
    norm = mpl.colors.BoundaryNorm(np.arange(-0.5,max_num), cmap.N) 


    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(np.rint(canv), cmap=cmap, vmin=-0.5,vmax=int(np.nanmax(canv))+0.5)
    
    ax.add_patch(plt.Circle((size//2,size//2),rad,facecolor='none',
                 edgecolor='yellow',alpha=1,lw=3))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    cbar = plt.colorbar(im,ticks=range(0,max_num+1),cax=cax)
    cbar.ax.set_ylabel('Coverage count')
    
    
    
    if teeth_locs is not None:
        width=1.4
        y1 = (size//2-rad)//4
        x1 = size//2

        x2 = size//2
        y2 = size//2-rad-size//50
        
        for deg in teeth_locs:
            rx1,ry1 = rot_point(x1,y1,deg,size)
            rx2,ry2 = rot_point(x2,y2,deg,size)

            ax.arrow(rx1,ry1,rx2-rx1,ry2-ry1,length_includes_head=True,width=width,head_width=4*width,head_length = 6*width,fill=True,color='black')
            ax.annotate(f'{deg:.1f}\xb0', xy=(rot_point(x1,y1-10,deg,size)))

    return fig

def histo(canv,size,rad):
    fig, ax = plt.subplots(figsize=(6, 4))

    xx, yy = np.mgrid[:size, :size]
    circle = (xx - size//2) ** 2 + (yy - size//2) ** 2

    #dat = np.where(circle<rad**2,canv,np.nan)
    
    #plt.imshow(np.nan_to_num(dat))
    #plt.show()
    #dat = np.nan_to_num(np.rint(np.ravel(dat)))
    dat = np.nan_to_num(np.rint(np.ravel(canv[circle<rad**2])))

    
    N, bins, patches = ax.hist(dat, bins=np.arange(-0.5, max(dat) + 1.5, 1),
                               edgecolor='black', linewidth=1, density=True,)#

    cols = ['white',
     '#1f77b4',
     '#d62728',
     '#e377c2',
     '#17becf',
     '#9467bd',
     '#ff7f0e',
     '#7f7f7f',
     '#2ca02c',
     '#bcbd22',
     '#1f77b4',
     '#d62728',
     '#e377c2',
     '#17becf',
     '#9467bd',
     '#ff7f0e',
     '#7f7f7f',
     '#2ca02c',
     '#bcbd22']

    for i,patch in enumerate (patches):
        patch.set_facecolor(cols[i])
    
    ax.set_ylabel('Proportion of target pixels')
    ax.set_xlabel('Coverage count')

    return fig,dat

@st.experimental_memo(max_entries=20,show_spinner = False)  
def sim_arc(size = 500,rad = None, num_teeth=2,arc=True,arc_start=0,arc_stop=90,plus_reverse=False,_bar=None):

    if rad is None:
        rad = size//4
    
       #200
    thick = rad//5    #20

    plat=False

    canv = np.empty((size,size))
    canv.fill(np.nan)

    teeth_locs = []

    #giffing=False



    deg = arc_start

    #if reverse:
    #    deg = arc_stop

    if plus_reverse:
        arc_mult=2
    else:
        arc_mult=1

    k=0

    
        

    for j in range(arc_mult*num_teeth):
        
        if arc and plus_reverse and j >= num_teeth:
            reverse=True
        else:
            reverse=False
        
        
        if not arc:
                deg = (arc_stop-arc_start)*j/num_teeth
        
        teeth_locs.append(deg)
        
        for i, depth in enumerate (range(rad,-rad,-thick)):

            
            #print(j,deg,depth)
            canv = draw_bar(canv,size,deg,depth,1.1*thick,rad)
            if plat:
                canv = draw_plateau(canv,size,deg,depth,1.1*thick,rad)
            
            if arc:
                if not reverse:
                    deg += (arc_stop-arc_start)/(len(range(rad,-rad,-thick))*num_teeth)
                else:
                    deg += -(arc_stop-arc_start)/(len(range(rad,-rad,-thick))*num_teeth)
            
            #if giffing:
            #    arc_plot(canv,fname=f'{k}.png')
            #    k+=1
            
            if _bar is not None:
            
                prog = (i+1+(j*2*rad//thick))/(arc_mult*num_teeth*2*rad//thick)
            
                _bar.progress(min(1,prog))
            
    return canv, teeth_locs


st.title('Arc coverage sim')

with st.form("arc_params"):
   
    col1,col2 = st.columns(2)
    
    with col1:
        (arc_start,arc_stop) = st.slider('arc_length',min_value=0,max_value=360,value=(0,90),step = 5)
    
        arc = st.checkbox('Arc?',value=True)
        teeth_arrows=st.checkbox('Display beam/teeth arrows?',value=False)
    with col2:
    
        num_teeth = st.slider('number of teeth/beams',min_value=1,max_value=10,value=2,step = 1)
        
        plus_reverse=st.checkbox('Plus reverse arc? (doubles number of teeth)',value=False)
    
    
    submitted  = st.form_submit_button('Sim')#, on_click=sim(arc_start=arc_start,arc_stop=arc_stop))
    
if submitted:

    if not arc and plus_reverse:
        plus_reverse=False
        st.warning('arc not set, ignoring addional reverse arc')

    #def sim(arc_start,arc_stop):
    size = 480
    rad = round(size/3)
    
    
    
    
    with st.spinner('Constructing...'):
        my_bar = st.progress(0)    
        canv, teeth_locs = sim_arc(size=size,rad=rad,arc_start=arc_start,arc_stop=arc_stop,arc=arc,num_teeth=num_teeth,plus_reverse=plus_reverse,
        _bar=my_bar)
    
    disp_arrows = teeth_locs if teeth_arrows else None
    
    st.text("")
    
    if num_teeth == 1:
        beam_type0 = f'tooth' if arc else f'beam'
    else:
        beam_type0 = f'teeth' if arc else f'beams'
    
    #toothy = 'tooth' if num_teeth==1 else 'teeth'
    
    arc_type = 'arc' if arc else 'beam'
    
    #if num_teeth ==1 and not plus_reverse:
    #    beam_type = f'tooth' if arc else f'beam'
    #else:
    #    beam_type = f'teeth' if arc else f'beams'
    
    rev_text = ' plus reverse arc' if plus_reverse else ''
    
    
    
    desc_str = f"## {num_teeth} {beam_type0} {arc_start}\xb0 - {arc_stop}\xb0 {arc_type}{rev_text}:"
    
    if plus_reverse:
        b_locs0 = ', '.join([f'{x}\xb0' for x in teeth_locs[:len(teeth_locs)//2]])
        b_locs1 = ', '.join([f'{x}\xb0' for x in teeth_locs[len(teeth_locs)//2:]])
        
        t_str = f"\n  #### CCW {beam_type0} at {b_locs0}; CW {beam_type0} at {b_locs1}"
        
    else:
        b_locs0 = ', '.join([f'{x}\xb0' for x in teeth_locs])
        t_str = f"\n  #### {beam_type0.capitalize()} at {b_locs0}"
    
    
    st.markdown(desc_str + t_str)
    st.text("")
    st.text("")
    
    col1,col2 = st.columns(2)
    
    with col1:
        st.markdown('### Coverage plot')
        st.pyplot(arc_plot(canv,size,rad,disp_arrows))
    with col2:
        st.markdown('### Pixel coverage counts within target')
        fig,dat = histo(canv,size,rad)
        
        df = pd.Series(dat).astype(int).value_counts().sort_index()/len(dat)
        df = df.to_frame().T
        df.rename(index={0:'Pixel proportion'},inplace=True)
        st.text("")
        st.dataframe(df.style.format('{0:.2%}'))
        st.text("")
        st.text("")
        st.pyplot(fig )