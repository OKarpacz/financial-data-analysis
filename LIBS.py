import os
import pickle
import requests
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from bs4 import BeautifulSoup
from collections import Counter
from sklearn import svm,neighbors,model_selection
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
