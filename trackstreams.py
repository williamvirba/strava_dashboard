import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
from gpxcsv import gpxtolist
import numpy as np
import matplotlib.pyplot as plt
import haversine as hs
import matplotlib as mpl




#streamlit run pythonfile.py into terminal streamlit run trackstream.py

st.set_page_config(page_title="Track Stream", page_icon=":bar_chart:", layout="wide")



df = pd.DataFrame(gpxtolist("Evening_Runw.gpx"))



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



zliness = df['ele']
xliness = df['lon']
yliness = df['lat']
c=df["hr"]
s=df["Speed"]


tags =["Very Low" ,"Low" ,"Warm Up" ,"Fat Burn" ,"Build Fitness" ,"High Intensity" ,"Extreme" ,"Health Threat","Life Threat"]

myfeelbound=[46,60,80,120,140,160,180,195,220,260]
myfeelbounds=[0,46,60,80,120,140,160,180,195,220]
colourlist =["grey","#666666","blue" ,"green" ,"yellow" ,"orange" ,"red","brown","purple","black"]


assert len(myfeelbound)== len(colourlist)
cdmap = mpl.colors.ListedColormap(colourlist)
normd = mpl.colors.BoundaryNorm(boundaries=myfeelbound, ncolors=len(cdmap.colors)+1,extend="both",clip=False )


st.title(":bar_chart: Track Stream ")
st.markdown("##")


st.sidebar.header("Please Filter Here:")
speeds = st.sidebar.multiselect(
    "Select the Speeds:",
    options=df["Speed"].unique(),
    default=["Speed"].unique()
)

df_selection = df.query(
    "Speed == @speeds"
)




st.dataframe(df_selection.iloc[: , :20])




lls=range(0,len(myfeelbounds)-1)
#print(myfeelbound[1])
df["id"]=np.arange(len(df))
#figss = plt.figure()

fig = plt.figure(2,figsize = (16, 20))
ax1 = plt.subplot(212)


for i in lls :

    mask= ((df['hr'] >= myfeelbounds[i]) & (myfeelbounds[i+1] > df['hr']))

    col = (df.loc[mask]['hr'])
    #for s in df.loc[mask]:
    #colorr.append(colourlist[i])

    ax1.bar(df["id"][mask], df['hr'][mask], color = colourlist[i],width=2)


pass

ax2 = plt.subplot(211,projection ="3d")
plot=ax2.scatter(xliness,yliness, zliness, c=c,cmap=cdmap,s=s*5,norm=normd)
plt.colorbar(plot,spacing='proportional',label="Heart Rate scale")



st.write(fig)
plt.show()

























