import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("NBA Player Statistics Explorer")

st.markdown("""
This app performs simple webscraping of NBA players stats data
* **python libraries:** base64, pandas, streamlit, matplotlib, seaborn
* **data source:** [Basketball-Reference.com](https://www.basketball-reference.com/).
""")

st.sidebar.header("User Input Features")
selected_year = st.sidebar.selectbox("Year", list(reversed(range(2000, 2024))))

# Web scraping of NBA player stats data
@st.cache_data
def load_data(year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
    html = pd.read_html(url, header=0)
    df = html[0]
    raw = df.drop(df[df.Age == "Age"].index)  # Remove header row
    raw = raw.fillna(0)  # Fill NaN values with 0
    playerstats = raw.drop(["Rk"], axis=1)  # Drop unnecessary columns
    return playerstats
playerstats = load_data(selected_year)
print(playerstats.columns)

# Sidebar - Team Selection
sorted_unique_teams = sorted(playerstats["Team"].astype(str).unique())
selected_team = st.sidebar.multiselect("Team", sorted_unique_teams, sorted_unique_teams)

# Sidebar - Position Selection
unique_positions = ['C', 'PF', 'SF', 'PG', 'SG']
selected_position = st.sidebar.multiselect("Position", unique_positions, unique_positions)

# Filter data based on selections
df_selected_team = playerstats[(playerstats["Team"].isin(selected_team)) & (playerstats.Pos.isin(selected_position))]

st.header(f"Displaying Player Statistics for {selected_year}")
st.write("Data Dimension: {} rows and {} columns".format(df_selected_team.shape[0], df_selected_team.shape[1]))
st.dataframe(df_selected_team)

# Download NBA Player Stats Data
def fileDownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="nba_player_stats.csv">Download CSV File</a>'
    return href

st.markdown(fileDownload(df_selected_team), unsafe_allow_html=True)

# Heatmap
if st.button('Intercorellation Heatmap'):
    st.subheader('Interorrelation Matrix Heatmap')
    df_selected_team.to_csv('output.csv', index=False)
    df = pd.read_csv('output.csv')
    
    corr = df.select_dtypes(include=[float, int]).corr()
    mask = np.zeros(corr.shape)
    mask[np.triu_indices_from(mask)] = True  # Mask the upper triangle
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot(f)