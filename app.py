import streamlit as st
st.write("""
#Hello
""")

number=st.slider("Hey pick number",0,10)


print(number**2)

st.write("""
{number}
""".(number))
