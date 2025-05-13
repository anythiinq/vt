"""
hello! welcome to my vt clustering dashboard. this website was created with streamlit. my dashboard can: 
1. give a visual representation of my data
2. easily let you predict the vt of a cell.
have fun! 

author: krystal sun
date: 6 may 2025
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import streamlit as st

st.set_page_config(page_title="vt clustering dashboard", layout="wide")

# ------------------------------------------------------------------
# 1. data loading & caching
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(pattern: str = "data/sample*-data*.csv") -> pd.DataFrame:
    files = glob.glob(pattern)
    if not files:
        st.error("no sample*-data*.csv files found in the current folder ❗")
        st.stop()

    df_list: list[pd.DataFrame] = []
    for f in files:
        try:
            tmp = pd.read_csv(f)
            tmp["source_file"] = os.path.basename(f)
            df_list.append(tmp)
        except Exception as exc:
            st.warning(f"⚠️ could not read {f}: {exc}")

    if not df_list:
        st.error("all csv reads failed — check file formats ✨")
        st.stop()
    return pd.concat(df_list, ignore_index=True)

raw_df = load_data()

expected_cols = {"die", "wl", "block", "hours", "vt"}
missing = expected_cols.difference(raw_df.columns)
if missing:
    st.error(f"missing columns in csv: {', '.join(missing)}")
    st.stop()

# ------------------------------------------------------------------
# 2. sidebar controls
# ------------------------------------------------------------------
block_options = sorted(raw_df["block"].unique())
default_block = 45 if 45 in block_options else block_options[0]
sel_block = st.sidebar.selectbox("choose a physical block", block_options,
                                  index=block_options.index(default_block))

kmin, kmax = 2, 8
sel_k = st.sidebar.slider("number of clusters (k‑means)", kmin, kmax, 5)
st.sidebar.caption("adjust k to change the number of clusters in the graph. according to the elbow method, k = 5 is the optimal choice.")

# ------------------------------------------------------------------
# 3. data prep
# ------------------------------------------------------------------
block_df = raw_df[raw_df["block"] == sel_block].copy()

coords = block_df[["die", "wl"]].values
if len(block_df) < sel_k:
    st.error("selected block has fewer rows than k clusters 🐣")
    st.stop()

km = KMeans(n_clusters=sel_k, random_state=0, n_init=10)
block_df["cluster_id"] = km.fit_predict(coords)

palette = sns.color_palette("pastel", sel_k).as_hex()

st.title(f"vt data collection — block {sel_block}")

def two_line_caption(head: str, tail: str):
    st.caption(f"*{head}.*\n*{tail}*")

# ------------------------------------------------------------------
# 4. terminology & concept explanation
# ------------------------------------------------------------------
with st.expander("click to see some terms explained!"):
    st.markdown("""
    **vt (threshold voltage)**: the voltage at which a memory cell switches its state. it’s super important for deciding if the data is still readable or not.

    **die**: a physical chip inside the NAND package. each die contains lots of memory blocks.

    **wl (wordline)**: a horizontal line that selects a row of cells inside a block. like a street in a grid of memory cells.

    **block**: a group of memory cells. kind of like a neighborhood inside the chip.

    **hours**: the amount of time that has passed since the data was written (retention time). affects how stable the vt is.

    **cluster**: a group of cells that behave similarly. clustering helps us simplify and predict vt more easily.

    **k-means**: a machine learning algorithm that puts similar data into k groups based on physical features.

    **linear regression**: a model that draws a line through the data to predict vt based on things like die, wl, and hours.
    """)

# ------------------------------------------------------------------
# 5. 3‑d scatter
# ------------------------------------------------------------------
with st.container():
    st.subheader("physical geometry vs vt")
    fig3d = px.scatter_3d(block_df, x="die", y="wl", z="vt", color="cluster_id",
                          color_discrete_sequence=palette,
                          labels=dict(die="die", wl="wordline", vt="vt (mV)",
                                      cluster_id="cluster"),
                          title="die × wl × vt")
    st.plotly_chart(fig3d, use_container_width=True)
    two_line_caption(
        "in this graph, we plot block 45's cells according to their die, wl, and vt values" 
        "this graph helps us use machine learning to separate the cells into distinct clusters, making it easier for us to see which cells are similar to each other and which cells have similar vt values ",
        "this graph gives us important information to help us predict the vt of a cell based on its die and wl values"
    )

# ------------------------------------------------------------------
# 5. prediction with dropdown + reveal button 
# ------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("it's your turn! test a real vt prediction.")

valid_inputs = block_df[['die', 'wl', 'hours']].drop_duplicates().sort_values(['die', 'wl', 'hours'])
valid_inputs['label'] = valid_inputs.apply(lambda r: f"die {r.die}, wl {r.wl}, {r.hours} h", axis=1)
choice = st.sidebar.selectbox("pick an input with actual vt", valid_inputs['label'])

sel_row = valid_inputs[valid_inputs['label'] == choice].iloc[0]
d, w, h = int(sel_row.die), int(sel_row.wl), int(sel_row.hours)

# cluster + model
models = {}
for cid in block_df['cluster_id'].unique():
    sub = block_df[block_df['cluster_id'] == cid]
    X = sub[['die', 'wl', 'hours']]
    y = sub['vt']
    model = LinearRegression().fit(X, y)
    models[cid] = model

chosen_cluster = km.predict([[d, w]])[0]
model = models[chosen_cluster]
pred_vt = int(round(model.predict([[d, w, h]])[0]))

st.sidebar.success(f"predicted vt: {pred_vt} mV ✨")

if st.sidebar.button("click to reveal actual vt"):
    actual = block_df[(block_df['die'] == d) & 
                      (block_df['wl'] == w) & 
                      (block_df['hours'] == h)]['vt'].values[0]
    st.sidebar.info(f"🎯 actual vt: {actual:.0f} mV")

st.sidebar.caption("predicting an accurate vt value is important because it helps us understand how the cell will behave in real life. this is important for making sure our chips work well and last a long time. if we can predict the vt accurately, we can make better chips that are more reliable and efficient.")

# ------------------------------------------------------------------
# 6. sidebar info
# ------------------------------------------------------------------
with st.sidebar.expander("data overview"):
    st.write(f"**rows loaded:** {len(raw_df):,}")
    st.write(f"**current block rows:** {len(block_df):,}")
    st.write("**csv files detected:**")
    for f in glob.glob("sample*-data*.csv"):
        st.write("・", os.path.basename(f))