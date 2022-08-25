import seaborn as sns

import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
from gpxcsv import gpxtolist
import pandas as pd
#streamlit run pythonfile.py into terminal streamlit run trackstream.py

st.set_page_config(page_title="Track Stream", page_icon=":bar_chart:", layout="wide")



df = pd.DataFrame(gpxtolist('/Users/viliamvirba/Downloads/Evening_Run.gpx'))




import numpy as np
import matplotlib.pyplot as plt

import haversine as hs



def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    distance = hs.haversine(
        point1=(lat1, lon1),
        point2=(lat2, lon2),
        unit=hs.Unit.METERS
    )
    return np.round(distance, 2)


distances = [np.nan]

for i in range(len(df)):
    if i == 0:
        continue
    else:
        distances.append(haversine_distance(
            lat1=df.iloc[i - 1]['lat'],
            lon1=df.iloc[i - 1]['lon'],
            lat2=df.iloc[i]['lat'],
            lon2=df.iloc[i]['lon']
        ))

df['distance'] = distances
df['elevation_diff'] = df['ele'].diff()
df['cum_elevation'] = df['elevation_diff'].cumsum()
df['cum_distance'] = df['distance'].cumsum()


df['Date'] = df['time'].str[:10]
df['Time'] = df['time'].str[11:19]
df['Time'] = pd.to_datetime(df['Time'],format= '%H:%M:%S').dt.time
df['Times'] = pd.to_datetime(df['Time'],format= '%H:%M:%S').dt.second
df['Timem'] = pd.to_datetime(df['Time'],format= '%H:%M:%S').dt.minute
df['Timeh'] = pd.to_datetime(df['Time'],format= '%H:%M:%S').dt.hour
df['Timess'] = df['Times']+df['Timem']*60+df['Timem']*3600
df['Timed'] = df['Timess'].diff()

df['Speed'] = df['distance']/df['Timed']*3600/1000



print(df)
print(df.dtypes)

zliness = df['ele']
xliness = df['lon']
yliness = df['lat']
c=df["hr"]
s=df["Speed"]


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mplc



import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



tags =["Very Low" ,"Low" ,"Warm Up" ,"Fat Burn" ,"Build Fitness" ,"High Intensity" ,"Extreme" ,"Health Threat","Life Threat"]

myfeelbound=[46,60,80,120,140,160,180,195,220,260]
myfeelbounds=[0,46,60,80,120,140,160,180,195,220]
colourlist =["grey","#666666","blue" ,"green" ,"yellow" ,"orange" ,"red","brown","purple","black"]


assert len(myfeelbound)== len(colourlist)
cdmap = mpl.colors.ListedColormap(colourlist)
normd = mpl.colors.BoundaryNorm(boundaries=myfeelbound, ncolors=len(cdmap.colors)+1,extend="both",clip=False )


cmapw = plt.cm.jet  # define the colormap
# extract all colors from the .jet map
cmapwlist = [cmapw(i) for i in range(cmapw.N)]
# force the first color entry to be grey
cmapwlist[0] = (.5, .5, .5, 1.0)

# create the new map
cmapw = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', colourlist, len(colourlist))

# define the bins and normalize
bounds = np.linspace(0, len(myfeelbound), len(myfeelbound)+1)
norm = mpl.colors.BoundaryNorm(bounds, cmapw.N)

# make the scatter
#scat = ax.scatter(x, y, c=tag, s=np.random.randint(100, 500, 20),
                  #cmap=cmapw, norm=norm)

# create a second axes for the colorbar
#ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
#cb = plt.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
    #spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')











st.title(":bar_chart: Track Stream ")
st.markdown("##")

#fig=px.scatter_3d(df,x=xliness, y=yliness,z=zliness,color=c)
#st.write(fig)
#fig.show()

#st.dataframe(df)

st.sidebar.header("Please Filter Here:")
hrs = st.sidebar.multiselect(
    "Select the HRs:",
    options=df["hr"].unique(),
    default=df["hr"].unique()
)

df_selection = df.query(
    "hr == @hrs"
)
st.dataframe(df_selection)

import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components



lls=range(0,len(myfeelbounds)-1)
#print(myfeelbound[1])
df["id"]=np.arange(len(df))
#figss = plt.figure()

fig = plt.figure(2,figsize = (16, 20))
ax1 = plt.subplot(212)



colorr=[]
for i in lls :

    mask= ((df['hr'] >= myfeelbounds[i]) & (myfeelbounds[i+1] > df['hr']))

    col = (df.loc[mask]['hr'])
    #for s in df.loc[mask]:
    #colorr.append(colourlist[i])

    ax1.bar(df["id"][mask], df['hr'][mask], color = colourlist[i],width=2)

#df['color'] = colorr

pass

ax2 = plt.subplot(211,projection ="3d")
plot=ax2.scatter(xliness,yliness, zliness, c=c,cmap=cdmap,s=s*5,norm=normd)
plt.colorbar(plot,spacing='proportional',label="Heart Rate scale")



st.write(fig)
plt.show()
#print(df["color"])
#print(col)

#for i in lls :

    #mask= ((df['hr'] >= myfeelbound[i]) & (myfeelbound[i+1] > df['hr']))

    #col = (df.loc[mask]['hr'])

    #px.scatter_3d(df[mask],df["lon"][mask],df["lat"][mask],df["ele"][mask], color =df["color"][mask] ,width=2)


#pass






#sns.set(style = "darkgrid")
#plt.ion()
#fig = plt.figure(figsize = (16, 9))

#ax2 = plt.axes(projection ="3d")
#ax = fig.add_subplot(projection="3d")

#ax2.scatter(xliness,yliness, zliness, c=c,cmap=cdmap,s=s*5,norm=normd)
#plot=ax.scatter3d(xliness,yliness, zliness, color=c,cmap=cdmap,s=s*5,norm=normd)

#plt.colorbar(plot,spacing='proportional',label="Heart Rate scale")








#ax = plt.axes(projection ="3d")
#plot=px.scatter_3d(xliness,yliness, zliness, color=c,color_discrete_map=cdmap)

#plot.colorbar(plot,spacing='proportional',label="Heart Rate scale ")
#st.write(plot)









#fig_html = mpld3.fig_to_html(fig)
#components.html(fig_html, height=600)





#fig.colorbar(plot)
#ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
#cb = mpl.colorbar.ColorbarBase(ax2, cmap=cdmap, norm=norm,
    #spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')








#figs = plt.figure(figsize = (16, 9))
#ax = figs.add_subplot()
#plots=ax.scatter(y=c,x=df["time"], c=c,cmap=cdmap,s=s*5,marker="|")
#plt.show()




hrb=df.groupby(['hr'])["hr"].count()

dfpl=df['hr'].plot(kind="bar")
#plt.show()




#print(hrb)
hrb.plot(kind="bar")






#figs=plt.figure(figsize = (16, 9))
#ax = figs.add_axes([0,0,1,1])
#plotb=ax.bar(hrb[:1],hrb[:2],color="red")

#plt.show()
#st.plotly_chart(fig, use_container_width=True)

import altair as alt
from vega_datasets import data



#gg=alt.Chart(df).mark_circle().encode(
    #x=xliness,
    #y=yliness,

    #color=c,
#).interactive()


