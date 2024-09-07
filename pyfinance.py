from LIBS import *

#1 - BUY
#-1 - SELL
#0 - HOLD
def save_sp500_tickers():
    try:
        resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup = BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class': 'wikitable sortable'})

        if table is None:
            print("Table not found")
            return []

        print("Table found")

        tickers = []
        rows = table.findAll('tr')
        print(f"Number of rows found: {len(rows)}")

        for row in rows[1:]:
            cells = row.findAll('td')
            if cells:
                ticker = cells[0].text.strip().replace('.', '-')
                tickers.append(ticker)

        if not tickers:
            print("No tickers found")

        with open('sp500tickers.pickle', 'wb') as f:
            pickle.dump(tickers, f)

        print("Tickers saved:", tickers)
        return tickers

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        if os.path.exists('sp500tickers.pickle'):
            with open('sp500tickers.pickle', 'rb') as f:
                tickers = pickle.load(f)
        else:
            print("No ticker file found. Please reload S&P 500 tickers.")
            return

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2019, 6, 8)
    end = dt.datetime.now()

    for ticker in tickers:
        print(f"Processing {ticker}")
        file_path = f'stock_dfs/{ticker}.csv'
        if not os.path.exists(file_path):
            try:
                df = yf.download(ticker, start=start, end=end)
                df.reset_index(inplace=True)
                df.set_index("Date", inplace=True)
                df.to_csv(file_path)
                print(f"Saved data for {ticker}.")
            except Exception as e:
                print(f"Could not retrieve data for {ticker}: {e}")
        else:
            print(f'Already have data for {ticker}.')

def compile_data():
    with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)
    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/'+ticker+'.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')
        if count % 10 == 0:
            print(count)
    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')

def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
    df.set_index('Date', inplace=True)
    df_corr = df.pct_change().corr()

    print(df_corr.head())

    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)

    plt.xticks(rotation=90)
    heatmap.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()

def process_data_for_labels(ticker, hm_days=7):
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    if df.shape[0] < hm_days:
        raise ValueError(f"Not enough data to shift by {hm_days} days")

    for i in range(1, hm_days+1):
        column_name = f'{ticker}_{i}d'
        df[column_name] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
        print(f"Created column: {column_name}")

    df.fillna(0, inplace=True)
    return tickers, df

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > 0.029:
            return 1
        if col < -0.027:
            return -1
    return 0

def extract_futuresets(ticker, hm_days=7):
    tickers, df = process_data_for_labels(ticker, hm_days)
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold, *[df[f'{ticker}_{i}d'] for i in range(1, hm_days+1)]))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    Y = df['{}_target'.format(ticker)].values

    return X, Y, df

def do_ml(ticker):
    X, Y, df = extract_futuresets(ticker)

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.25)

    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    clf.fit(X_train, Y_train)

    confidence = clf.score(X_test, Y_test)
    print('Accuracy: ', confidence)
    predictions = clf.predict(X_test)
    predictions = [int(p) for p in predictions]
    print('Predicted spread:', Counter(predictions))

    return confidence

# do_ml('BAC')
# extract_futuresets('XOM')
# save_sp500_tickers()
# get_data_from_yahoo()
# compile_data()
# visualize_data()
