import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.figure_factory as ff
from vega_datasets import data
import pandas as pd


st.title("Second Page")

# simple area charts
st.subheader("simple area charts")
chart_data = pd.DataFrame(np.random.randn(20, 3),columns=['a', 'b', 'c'])
st.area_chart(
    chart_data, x = "a", y = ["b","c"],color=["#FF0000","#00FF00"]
)

# simple bar charts
st.subheader("simple bar charts")
source = data.barley()
st.bar_chart(source, x="variety", y="yield", color="site", horizontal=True)

# simple line charts
st.subheader("simple line charts")
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
st.line_chart(chart_data)

# simple scatter charts
st.subheader("simple scatter charts")
chart_data = pd.DataFrame(np.random.randn(20, 3),columns=['a', 'b', 'c'])
#加上size大小變化
chart_data = pd.DataFrame(np.random.randn(20, 3),columns=['col1', 'col2', 'col3'])
chart_data["col4"] = np.random.choice(["A","B","C"],20)

st.scatter_chart(chart_data,
                 x="col1",
                 y="col2",
                 size="col4",
                 color="col3")



st.header("3D plot")
x, y, z = np.random.multivariate_normal(np.array([0, 0, 0]), np.eye(3), 400).transpose()
trace1 = go.Scatter3d(
    x=x, y=y, z=z, mode="markers",
    marker=dict(size=12, color=z, colorscale="Viridis", opacity=0.8)
)
data = [trace1]
layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
fig = go.Figure(data=data, layout=layout)
st.plotly_chart(fig)

st.header("Fancy density plot")
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2
hist_data = [x1, x2, x3]
group_labels = ["Group 1", "Group 2", "Group 3"]
fig = ff.create_distplot(hist_data, group_labels, bin_size=[0.1, 0.25, 0.5])
fig.update_layout(bargap=0.1)  # Ensure bargap is within the valid range
st.plotly_chart(fig)
