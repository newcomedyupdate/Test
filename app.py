import streamlit as st
st.write("""
#Hello
""")

number=st.slider("Hey pick number",0,10)


st.write("Square of number",number**2)
import pandas as pd
st.write("Hey")
file=st.file_uploader("Please upload CSV here")


data = {'name': ['nick', 'david', 'joe', 'ross'],
		'age': ['5', '10', '7', '6']}
df = pd.DataFrame.from_dict(data)


df=pd.read_csv("file")

st.write(df)
