# 04_Src/sanity_check.py

def main():
    print("Testing imports...")

    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import yfinance as yf
        import torch
        import streamlit as st
        import shap
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        print("All core libraries imported successfully!")
    except Exception as e:
        print("Import error:")
        print(e)

if __name__ == "__main__":
    main()
