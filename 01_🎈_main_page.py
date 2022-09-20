#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 18:09:54 2022

@author: alessandro
"""

import streamlit as st

st.title("MY FIRST ML APP")

my_text = st.text("A random version 2 of text string")

my_button = st.button("Run ML computation")

if my_button:
    st.title("The model is running ...")
    
# boston = load_boston()
# df = pd.DataFrame(boston.data, columns = boston.feature_names)
# st.dataframe(df)
