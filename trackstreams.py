import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
from gpxcsv import gpxtolist
import numpy as np
import matplotlib.pyplot as plt
import haversine as hs
import matplotlib as mpl
import tilemapbase



#streamlit run pythonfile.py into terminal streamlit run trackstream.py



df = pd.DataFrame(gpxtolist("99_MMM_2022.gpx"))

st.set_page_config(page_title=df["name"][0],page_icon=":bar_chart:", layout="wide")







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




st.title(df["name"][0])
st.markdown("##")

st.sidebar.header("Please Filter Here:")
hrs = st.sidebar.multiselect(
    "Select the HRs:",
    options=df["hr"].unique(),
    default=df["hr"].unique()
)

df_selection = df.query(
    "hr == @hrs"
)








average_hr = (df_selection["hr"].mean())
sd_hr = (df_selection["hr"].std())
tot_dis = (df_selection["cum_distance"].max())
tot_el = (df_selection["cum_elevation"].max())

f_column,s_column= st.columns(2)

with f_column:
    st.subheader("Average HR [bpm] :")
    st.subheader(f"{average_hr}")

with s_column:
    st.subheader("sd HR [bpm] :")
    st.subheader(f"{sd_hr}")

t_column,f_column= st.columns(2)    
    
with t_column:
    st.subheader("Total Distance [km] :")
    st.subheader(f"{tot_dis/1000}")
    
    
with f_column:
    st.subheader("Total Elevation diff [m] :")
    st.subheader(f"{tot_el}")    

st.markdown("""---""")


bounding_box = [df["lat"].min(), df["lat"].max(), df["lon"].min(), df["lon"].max()]
print(bounding_box)


path = [tilemapbase.project(x,y) for x,y in zip(df["lon"], df["lat"])]
x, y = zip(*path)


tilemapbase.start_logging()
tilemapbase.init(create=True)
t = tilemapbase.tiles.build_OSM()

my_office = (21.261074,48.718155)

degree_range = 0.05

extent = tilemapbase.Extent.from_lonlat(my_office[0] - degree_range, my_office[0] + degree_range,
                  my_office[1] - degree_range, my_office[1] + degree_range)
extent = extent.to_aspect(1.0)


# On my desktop, DPI gets scaled by 0.75
figas, ax = plt.subplots(figsize=(10, 10), dpi=100)

ax.xaxis.set_visible(True)
ax.yaxis.set_visible(True)

plotter = tilemapbase.Plotter(extent, t, width=600)
plotter.plot(ax, t)

#x, y = tilemapbase.project(*my_office)
#ax.scatter(x,y, marker=".", color="black", linewidth=20)

myfeelbound=[46,60,80,120,140,160,180,195,220,260]
myfeelboundsp=[0,46,60,80,120,140,160,180,195,220]
myfeelbounds=[0,80,120,140,150,160,170,180,195,220]
colourlist =["grey","#666666","blue" ,"green" ,"yellow" ,"orange" ,"red","brown","purple","black"]


assert len(myfeelbounds)== len(colourlist)
cdmap = mpl.colors.ListedColormap(colourlist)
normd = mpl.colors.BoundaryNorm(boundaries=myfeelbounds, ncolors=len(cdmap.colors)+1,extend="both",clip=False )



plot=ax.scatter(x,y,c=c,cmap=cdmap,s=df["Speed"]/10,norm=normd)
plt.colorbar(plot,spacing='proportional',label="W Heart Rate scale")
plt.title('MAP', loc='center')
st.write(figas)


figa=px.scatter_3d(df,x=xliness, y=yliness,z=zliness,color=c,color_continuous_scale="thermal",width=800, height=400)

figa.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="LightSteelBlue",
)
#st.write(figa)




lls=range(0,len(myfeelbounds)-1)
#print(myfeelbound[1])
df["id"]=np.arange(len(df))
#figss = plt.figure()

fig = plt.figure(2,figsize = (16, 20))
ax1 = plt.subplot(211)


for i in lls :

    mask= ((df['hr'] >= myfeelbounds[i-1]) & (myfeelbounds[i] > df['hr']))

    col = (df.loc[mask]['hr'])
    #for s in df.loc[mask]:
    #colorr.append(colourlist[i])

    ax1.bar(df["cum_distance"][mask], df['hr'][mask], color = colourlist[i],width=7)


pass

ax2 = plt.subplot(212,projection ="3d")
plot=ax2.scatter(xliness,yliness, zliness, c=c,cmap=cdmap,s=s*5,norm=normd)
plt.colorbar(plot,spacing='proportional',label="Heart Rate scale")



st.write(fig)
plt.show()

st.dataframe(df_selection.iloc[: , :20])























