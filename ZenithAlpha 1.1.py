# Importiere benötigte Module.  
import pandas as pd  # Importiert das Modul pandas zur Datenmanipulation und -analyse.  
import numpy as np  # Importiert das Modul numpy für numerische Berechnungen.  
import tkinter as tk  # Importiert das Modul tkinter zur Erstellung grafischer Benutzeroberflächen (GUI).  
from tkinter import messagebox  # Importiert die Funktion messagebox von tkinter zur Anzeige von Pop-up-Nachrichten.  
from PIL import Image, ImageTk  # Importiert die Module Image und ImageTk von Pillow, um Bilder zu laden und in der GUI darzustellen.  
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Importiert FigureCanvasTkAgg, um Matplotlib-Grafiken in tkinter einzubetten.  
import matplotlib.pyplot as plt  # Importiert pyplot von Matplotlib, um Diagramme zu erstellen.  
import requests  # Importiert das Modul requests, um HTTP-Anfragen (z. B. an APIs) zu senden.  
from datetime import datetime  # Importiert datetime zur Arbeit mit Datums- und Zeitangaben.  
import os  # Importiert das Modul os, um mit dem Dateisystem (z. B. Datei- und Pfadoperationen) zu arbeiten.  
from sklearn.ensemble import RandomForestRegressor  # Importiert den RandomForestRegressor aus scikit-learn für maschinelles Lernen.  
from sklearn.model_selection import train_test_split  # Importiert train_test_split zur Aufteilung von Datensätzen in Trainings- und Testdaten.  
from sklearn.metrics import mean_squared_error  # Importiert die Funktion mean_squared_error zur Bewertung von Vorhersagemodellen.  

# Projekt: ZenithAlpha von Lukas Völzing. Datum: März 2025. Unternehmen: Linoz Developments.  

# Diese Zeile enthält Projektdetails und dient als Information.  

# Programmversion 1.1. ========================================================  

# Willkommensmeldung.  

print()
print("Welcome by ZenithAlpha from Linoz Developments. Version 1.1. Developer: Lukas Voelzing")  # Gibt eine englische Begrüßung aus.  

# =============================================================================  
# DATENABRUF MIT ALPHA VANTAGE (TECHNISCH & FUNDAMENTAL).  
# =============================================================================  

def fetch_data_from_alpha_vantage(ticker, api_key):  # Definiert die Funktion zum Abruf historischer Kursdaten von Alpha Vantage.  
    try:  # Beginnt einen Try-Block zur Fehlerbehandlung.  
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}'  # Erstellt die URL für die API-Anfrage mit Ticker und API-Schlüssel.  
        response = requests.get(url)  # Sendet eine GET-Anfrage an die API.  
        data = response.json()  # Wandelt die API-Antwort in ein JSON-Format um.  

        if 'Time Series (Daily)' not in data:  # Prüft, ob der Schlüssel für die täglichen Zeitreihen in den Daten vorhanden ist.  
            print()  # Gibt eine leere Zeile in der Konsole aus.  
            print(f"No historical data available for: {ticker}.")  # Gibt eine Fehlermeldung aus, falls keine Daten gefunden wurden.  
            return None  # Beendet die Funktion und gibt None zurück, wenn keine Daten vorliegen.  

        time_series = data['Time Series (Daily)']  # Speichert die Zeitreihendaten aus dem JSON in der Variable time_series.  
        df = pd.DataFrame.from_dict(time_series, orient='index')  # Wandelt das Dictionary in einen Pandas DataFrame um, wobei die Keys als Index genutzt werden.  
        df = df.rename(columns={  # Bennent die Spalten des DataFrames um, um sie besser lesbar zu machen.  
            '1. open': 'Open',  # Benennt die Spalte für den Eröffnungskurs um.  
            '2. high': 'High',  # Benennt die Spalte für den Tageshochkurs um.  
            '3. low': 'Low',  # Benennt die Spalte für den Tagestiefkurs um.  
            '4. close': 'Close',  # Benennt die Spalte für den Schlusskurs um.  
            '5. volume': 'Volume'  # Benennt die Spalte für das Handelsvolumen um.  
        })
        df.index = pd.to_datetime(df.index)  # Wandelt den Index des DataFrames in Datetime-Objekte um.  
        df = df.sort_index(ascending=True)  # Sortiert den DataFrame chronologisch in aufsteigender Reihenfolge.  
        print()  # Gibt eine leere Zeile in der Konsole aus.  
        print(f"Historical data retrieved for: {ticker}")  # Gibt eine Erfolgsmeldung aus, dass die Daten abgerufen wurden.  
        return df  # Gibt den DataFrame mit den historischen Daten zurück.  
    except Exception as e:  # Fängt alle Fehler, die im Try-Block auftreten, ab.  
        print()  # Gibt eine leere Zeile in der Konsole aus.  
        print(f"Error retrieving data from Alpha Vantage: {str(e)}")  # Gibt eine Fehlermeldung mit dem Fehlertext aus.  
        return None  # Gibt None zurück, falls ein Fehler auftritt.  

def evaluate_stock_fundamentals_av(ticker, api_key):  # Definiert die Funktion zur Bewertung fundamentaler Aktienkennzahlen mit Alpha Vantage.  
    try:  # Beginnt einen Try-Block zur Fehlerbehandlung.  
        url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}'  # Erstellt die URL für den Abruf fundamentaler Daten.  
        response = requests.get(url)  # Sendet eine GET-Anfrage an die API.  
        info = response.json()  # Wandelt die Antwort in ein JSON-Format um.  

        if not info or "Symbol" not in info:  # Prüft, ob die Daten leer sind oder der Schlüssel "Symbol" fehlt.  
            print()  # Gibt eine leere Zeile in der Konsole aus.  
            print(f"No fundamental data available for: {ticker}.")  # Gibt eine Fehlermeldung aus, wenn keine fundamentalen Daten vorhanden sind.  
            return None  # Gibt None zurück, falls fundamentale Daten nicht gefunden werden.  

        peratio = info.get("PERatio", None)  # Holt den Wert für das Kurs-Gewinn-Verhältnis (PERatio) aus den Daten.  
        price_to_sales = info.get("PriceToSalesRatio", None)  # Holt den Wert für das Kurs-Umsatz-Verhältnis aus den Daten.  
        dividend_yield = info.get("DividendYield", None)  # Holt den Wert für die Dividendenrendite aus den Daten.  

        if peratio is not None and price_to_sales is not None:  # Prüft, ob beide Kennzahlen vorhanden sind.  
            try:  # Beginnt einen inneren Try-Block zur Fehlerbehandlung bei der Umwandlung.  
                peratio = float(peratio)  # Konvertiert das PERatio in einen Float-Wert.  
                price_to_sales = float(price_to_sales)  # Konvertiert das Price-to-Sales-Verhältnis in einen Float-Wert.  
            except:  # Falls die Umwandlung fehlschlägt:  
                peratio = None  # Setzt peratio auf None.  
                price_to_sales = None  # Setzt price_to_sales auf None.  

            if peratio is not None and price_to_sales is not None:  # Überprüft erneut, ob die Umwandlung erfolgreich war.  
                if peratio < 20 and price_to_sales < 2:  # Bewertet, ob beide Kennzahlen in einem attraktiven Bereich liegen.  
                    recommendation = "Attractively valued – potentially a good investment opportunity!"  # Setzt die Empfehlung als attraktiv.  
                else:  # Falls die Kennzahlen nicht im attraktiven Bereich liegen:  
                    recommendation = "Rather moderately to highly valued."  # Setzt die Empfehlung als moderat bis hoch bewertet.  
            else:  # Falls eine der Umwandlungen fehlgeschlagen ist:  
                recommendation = "Not all fundamental metrics are convertible."  # Gibt an, dass nicht alle Kennzahlen konvertiert werden konnten.  
        else:  # Falls eine der Kennzahlen überhaupt nicht vorhanden ist:  
            recommendation = "Not all fundamental metrics are available."  # Gibt an, dass nicht alle erforderlichen Kennzahlen verfügbar sind.  

        result = {  # Erstellt ein Dictionary, um die Ergebnisse zu speichern.  
            "PERatio": peratio,  # Speichert das Kurs-Gewinn-Verhältnis.  
            "PriceToSales": price_to_sales,  # Speichert das Kurs-Umsatz-Verhältnis.  
            "DividendYield": dividend_yield,  # Speichert die Dividendenrendite.  
            "recommendation": recommendation  # Speichert die Investment-Empfehlung.  
        }
        return result  # Gibt das Dictionary mit den fundamentalen Kennzahlen zurück.  
    except Exception as e:  # Fängt Fehler ab, die im Try-Block auftreten.  
        print(f"Error in fundamental analysis: {str(e)}")  # Gibt eine Fehlermeldung mit dem Fehlertext aus.  
        return None  # Gibt None zurück, falls ein Fehler auftritt.  

# =============================================================================  
# TECHNISCHE INDIKATOREN & BACKTESTING.  
# =============================================================================  

def calculate_sma(data, window=50):  # Definiert die Funktion zur Berechnung des einfachen gleitenden Durchschnitts (SMA).  
    data[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()  # Berechnet den SMA für die 'Close'-Preise über das angegebene Fenster und speichert ihn in einer neuen Spalte.  
    return data  # Gibt den DataFrame mit dem neuen SMA zurück.  

def calculate_rsi(data, window=14):  # Definiert die Funktion zur Berechnung des Relative Strength Index (RSI).  
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')  # Konvertiert die 'Close'-Preise in numerische Werte und ersetzt ungültige Werte mit NaN.  
    data = data.dropna(subset=['Close'])  # Entfernt alle Zeilen, in denen 'Close' NaN ist.  
    delta = data['Close'].diff(1)  # Berechnet die tägliche Differenz der Schlusskurse.  
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()  # Berechnet den gleitenden Durchschnitt der Gewinne über das Fenster.  
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()  # Berechnet den gleitenden Durchschnitt der Verluste über das Fenster.  
    rs = gain / loss  # Berechnet das Verhältnis von durchschnittlichen Gewinnen zu durchschnittlichen Verlusten.  
    data['RSI'] = 100 - (100 / (1 + rs))  # Berechnet den RSI-Wert und speichert ihn in einer neuen Spalte.  
    return data  # Gibt den DataFrame mit der RSI-Spalte zurück.  

def calculate_macd(data):  # Definiert die Funktion zur Berechnung des Moving Average Convergence Divergence (MACD).  
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()  # Berechnet den MACD als Differenz zweier exponentieller gleitender Durchschnitte.  
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()  # Berechnet die Signallinie des MACD als exponentiell gleitenden Durchschnitt des MACD.  
    return data  # Gibt den DataFrame mit den MACD-Werten zurück.  

def calculate_bollinger_bands(data, window=20):  # Definiert die Funktion zur Berechnung der Bollinger-Bänder.  
    data['Bollinger_Middle'] = data['Close'].rolling(window=window).mean()  # Berechnet den mittleren Wert (gleitender Durchschnitt) als Basislinie.  
    data['Bollinger_Upper'] = data['Bollinger_Middle'] + (data['Close'].rolling(window=window).std() * 2)  # Berechnet das obere Bollinger-Band als mittleren Wert plus zweimal die Standardabweichung.  
    data['Bollinger_Lower'] = data['Bollinger_Middle'] - (data['Close'].rolling(window=window).std() * 2)  # Berechnet das untere Bollinger-Band als mittleren Wert minus zweimal die Standardabweichung.  
    return data  # Gibt den DataFrame mit den Bollinger-Bändern zurück.  

def backtest_strategy(data):  # Definiert die Funktion zum Backtesting einer Handelsstrategie auf Basis technischer Indikatoren.  
    data['Signal'] = np.where(  # Setzt ein Kaufsignal (1) oder kein Signal (0) in einer neuen Spalte basierend auf bestimmten Bedingungen.  
        (data['RSI'] < 30) & (data['MACD'] > data['MACD_Signal']) &  # Bedingung: RSI unter 30 und MACD über der Signallinie.  
        (data['Close'] > data['SMA_50']) & (data['Close'] < data['Bollinger_Lower']),  # Bedingung: Schlusskurs über dem 50-Tage-SMA und unter dem unteren Bollinger-Band.  
        1, 0  # Setzt 1 für ein Kaufsignal und 0, falls die Bedingungen nicht erfüllt sind.  
    )
    data['Returns'] = data['Close'].pct_change() * data['Signal'].shift(1)  # Berechnet die täglichen prozentualen Renditen multipliziert mit dem vorherigen Handelssignal.  
    data['Portfolio_Value'] = (1 + data['Returns']).cumprod() * 1000  # Berechnet den kumulierten Portfoliowert, ausgehend von einem Startkapital von 1000.  
    return data  # Gibt den DataFrame mit den Backtesting-Ergebnissen zurück.  

# =============================================================================  
# MASCHINELLES LERNEN FÜR QUANTITATIVE STRATEGIEN.  
# =============================================================================  

def train_ml_model(data):  # Definiert die Funktion zum Trainieren eines Machine-Learning-Modells für quantitative Strategien.  
    data['Return'] = data['Close'].pct_change().shift(-1)  # Berechnet die zukünftige Rendite, indem die prozentuale Änderung der Schlusskurse um einen Tag verschoben wird.  
    data = data.dropna().copy()  # Entfernt Zeilen mit fehlenden Werten und erstellt eine Kopie des DataFrames.  

    if 'SMA_50' not in data.columns:  # Prüft, ob der 50-Tage-SMA bereits berechnet wurde.  
        data = calculate_sma(data)  # Berechnet den 50-Tage-SMA, falls nicht vorhanden.  
    if 'RSI' not in data.columns:  # Prüft, ob der RSI bereits berechnet wurde.  
        data = calculate_rsi(data)  # Berechnet den RSI, falls nicht vorhanden.  
    if 'MACD' not in data.columns or 'MACD_Signal' not in data.columns:  # Prüft, ob MACD und seine Signallinie bereits berechnet wurden.  
        data = calculate_macd(data)  # Berechnet den MACD, falls nicht vorhanden.  
    if 'Bollinger_Upper' not in data.columns or 'Bollinger_Lower' not in data.columns:  # Prüft, ob die Bollinger-Bänder bereits berechnet wurden.  
        data = calculate_bollinger_bands(data)  # Berechnet die Bollinger-Bänder, falls nicht vorhanden.  

    features = ['Close', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Upper', 'Bollinger_Lower']  # Definiert die Liste der Merkmale (Features) für das Modell.  
    X = data[features]  # Extrahiert die Merkmale aus dem DataFrame.  
    y = data['Return']  # Extrahiert die Zielvariable (zukünftige Rendite) aus dem DataFrame.  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # Teilt die Daten in Trainings- und Testdaten (20 % Testdaten), ohne die Reihenfolge zu mischen.  
    model = RandomForestRegressor(n_estimators=100, random_state=42)  # Initialisiert einen RandomForestRegressor mit 100 Bäumen und festem Zufallszustand.  
    model.fit(X_train, y_train)  # Trainiert das Modell mit den Trainingsdaten.  

    predictions = model.predict(X_test)  # Nutzt das Modell, um Vorhersagen für die Testdaten zu generieren.  
    mse = mean_squared_error(y_test, predictions)  # Berechnet den mittleren quadratischen Fehler (MSE) der Vorhersagen.  
    print(f"ML-Modell Mean Squared Error: {mse}")  # Gibt den MSE in der Konsole aus.  

    data['ML_Prediction'] = model.predict(X[features])  # Speichert die ML-Vorhersagen für alle Daten in einer neuen Spalte.  
    return data, model  # Gibt den aktualisierten DataFrame und das trainierte Modell zurück.  

def backtest_quant_strategy(data):  # Definiert die Funktion zum Backtesting einer quantitativen Strategie, die ML und technische Indikatoren kombiniert.  
    ml_threshold = 0.001  # Definiert einen Schwellenwert für die ML-Vorhersage.  
    data['Quant_Signal'] = np.where(  # Erzeugt ein quant. Handelssignal basierend auf ML-Vorhersage oder technischen Bedingungen.  
        (data['ML_Prediction'] > ml_threshold) |  # Signal, wenn die ML-Vorhersage den Schwellenwert überschreitet.  
        ((data['RSI'] < 30) & (data['MACD'] > data['MACD_Signal']) &  # Oder wenn technische Indikatoren ein Kaufsignal geben: RSI unter 30 und MACD über der Signallinie.  
         (data['Close'] > data['SMA_50']) & (data['Close'] < data['Bollinger_Lower'])),  # und Schlusskurs über SMA_50 aber unter dem unteren Bollinger-Band liegt.  
        1, 0  # Setzt 1 für ein Kaufsignal, sonst 0.  
    )
    data['Quant_Returns'] = data['Close'].pct_change() * data['Quant_Signal'].shift(1)  # Berechnet die quantitativen täglichen Renditen basierend auf dem Signal.  
    data['Quant_Portfolio'] = (1 + data['Quant_Returns']).cumprod() * 1000  # Berechnet den Portfoliowert der quantitativen Strategie ausgehend von 1000 Einheiten Startkapital.  
    return data  # Gibt den DataFrame mit den quantitativen Backtesting-Ergebnissen zurück.  

# =============================================================================  
# FUNKTIONEN FÜR DIE GUI.  
# =============================================================================  

def analyze_and_plot():  # Definiert die Funktion zur Analyse und Darstellung technischer Analysen inklusive Backtesting in der GUI.  
    ticker = entry_ticker.get().upper()  # Liest den eingegebenen Aktien-Ticker aus und wandelt ihn in Großbuchstaben um.  
    if not ticker:  # Prüft, ob kein Ticker eingegeben wurde.  
        messagebox.showerror("Error", "Please enter a valid ticker symbol!")  # Zeigt eine Fehlermeldung an, wenn kein Ticker vorhanden ist.  
        return  # Beendet die Funktion, falls kein Ticker eingegeben wurde.  
    api_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXX"  # Setzt den API-Schlüssel für Alpha Vantage. Dieser ist individuell. Account nötig.  
    data = fetch_data_from_alpha_vantage(ticker, api_key)  # Ruft die historischen Daten für den eingegebenen Ticker ab.  
    if data is not None:  # Prüft, ob die Daten erfolgreich abgerufen wurden.  
        data = calculate_sma(data)  # Berechnet den 50-Tage-SMA und fügt ihn den Daten hinzu.  
        data = calculate_rsi(data)  # Berechnet den RSI und fügt ihn den Daten hinzu.  
        data = calculate_macd(data)  # Berechnet den MACD und fügt ihn den Daten hinzu.  
        data = calculate_bollinger_bands(data)  # Berechnet die Bollinger-Bänder und fügt sie den Daten hinzu.  
        data = backtest_strategy(data)  # Führt das Backtesting der technischen Strategie durch.  
        plot_data(data, title="Technical Analysis & Backtest")  # Zeichnet die Ergebnisse in einem Diagramm in der GUI.  
        display_investment_opportunities(data)  # Zeigt eine Meldung zu Investitionsmöglichkeiten basierend auf dem Signal an.  
    else:  # Falls die Daten nicht abgerufen werden konnten:  
        messagebox.showerror("Error", "Error retrieving the data.")  # Zeigt eine Fehlermeldung in der GUI an.  

def analyze_fundamentals():  # Definiert die Funktion zur Analyse fundamentaler Kennzahlen in der GUI.  
    ticker = entry_ticker.get().upper()  # Liest den eingegebenen Aktien-Ticker aus und wandelt ihn in Großbuchstaben um.  
    if not ticker:  # Prüft, ob kein Ticker eingegeben wurde.  
        messagebox.showerror("Error", "Please enter a valid ticker symbol!")  # Zeigt eine Fehlermeldung an, wenn kein Ticker vorhanden ist.  
        return  # Beendet die Funktion, falls kein Ticker eingegeben wurde.  
    api_key = "YYYYYYYYYYYY"  # Setzt den API-Schlüssel für Alpha Vantage. Dieser ist individuell. Account nötig.   
    result = evaluate_stock_fundamentals_av(ticker, api_key)  # Ruft fundamentale Kennzahlen für den Ticker ab.  
    if result is not None:  # Prüft, ob fundamentale Daten erfolgreich abgerufen wurden.  
        info_str = (f"PERatio: {result['PERatio']}\n"  # Erstellt einen formatierten String mit dem PERatio.  
                    f"PriceToSales: {result['PriceToSales']}\n"  # Fügt den Price-to-Sales-Wert hinzu.  
                    f"DividendYield: {result['DividendYield']}\n\n"  # Fügt die Dividendenrendite hinzu.  
                    f"Recommendation: {result['recommendation']}")  # Fügt die Investment-Empfehlung hinzu.  
        messagebox.showinfo("fundamental analysis", info_str)  # Zeigt die fundamentalen Kennzahlen in einer Informationsbox an.  
    else:  # Falls fundamentale Daten nicht abgerufen werden konnten:  
        messagebox.showerror("Error", "Error in fundamental analysis.")  # Zeigt eine Fehlermeldung in der GUI an.  

def analyze_quant_strategy():  # Definiert die Funktion zur Analyse der quantitativen Strategie (ML und technisch) in der GUI.  
    ticker = entry_ticker.get().upper()  # Liest den eingegebenen Aktien-Ticker aus und wandelt ihn in Großbuchstaben um.  
    if not ticker:  # Prüft, ob kein Ticker eingegeben wurde.  
        messagebox.showerror("Error", "Please enter a valid ticker symbol!")  # Zeigt eine Fehlermeldung an, wenn kein Ticker vorhanden ist.  
        return  # Beendet die Funktion, falls kein Ticker eingegeben wurde.  
    api_key = "ZZZZZZZZZZZZ"  # Setzt den API-Schlüssel für Alpha Vantage. Dieser ist individuell. Account nötig.   
    data = fetch_data_from_alpha_vantage(ticker, api_key)  # Ruft die historischen Daten für den eingegebenen Ticker ab.  
    if data is not None:  # Prüft, ob die Daten erfolgreich abgerufen wurden.  
        data = calculate_sma(data)  # Berechnet den 50-Tage-SMA und fügt ihn den Daten hinzu.  
        data = calculate_rsi(data)  # Berechnet den RSI und fügt ihn den Daten hinzu.  
        data = calculate_macd(data)  # Berechnet den MACD und fügt ihn den Daten hinzu.  
        data = calculate_bollinger_bands(data)  # Berechnet die Bollinger-Bänder und fügt sie den Daten hinzu.  
        data, model = train_ml_model(data)  # Trainiert ein ML-Modell und speichert die Vorhersagen in den Daten.  
        data = backtest_quant_strategy(data)  # Führt das Backtesting der quantitativen Strategie durch.  
        plot_data(data, title="Quantitative Strategy (ML & Technical)")  # Zeichnet die quantitativen Strategieergebnisse in einem Diagramm in der GUI.  
        display_quant_investment_opportunities(data)  # Zeigt eine Meldung zu quantitativen Investitionsmöglichkeiten an.  
    else:  # Falls die Daten nicht abgerufen werden konnten:  
        messagebox.showerror("Error", "Error retrieving the data.")  # Zeigt eine Fehlermeldung in der GUI an.  

def plot_data(data, title="Analyse"):  # Definiert die Funktion zur Darstellung der Analyseergebnisse als Diagramm in der GUI.  
    fig, ax = plt.subplots(figsize=(10, 6))  # Erstellt eine Matplotlib-Figur und Achse mit der angegebenen Größe.  
    ax.plot(data.index, data['Close'], label='Schlusskurs', color='blue')  # Zeichnet den Schlusskurs als blaue Linie.  
    if 'SMA_50' in data.columns:  # Prüft, ob der 50-Tage-SMA in den Daten vorhanden ist.  
        ax.plot(data.index, data['SMA_50'], label='50-Tage-SMA', color='red')  # Zeichnet den 50-Tage-SMA als rote Linie.  
    if 'MACD' in data.columns and 'MACD_Signal' in data.columns:  # Prüft, ob MACD und die MACD-Signallinie vorhanden sind.  
        ax.plot(data.index, data['MACD'], label='MACD', color='green')  # Zeichnet den MACD als grüne Linie.  
        ax.plot(data.index, data['MACD_Signal'], label='MACD Signal', color='orange')  # Zeichnet die MACD-Signallinie als orange Linie.  
    if 'Bollinger_Upper' in data.columns and 'Bollinger_Lower' in data.columns:  # Prüft, ob die Bollinger-Bänder vorhanden sind.  
        ax.plot(data.index, data['Bollinger_Upper'], label='Bollinger Upper', color='purple', linestyle='--')  # Zeichnet das obere Bollinger-Band als gestrichelte, lila Linie.  
        ax.plot(data.index, data['Bollinger_Lower'], label='Bollinger Lower', color='purple', linestyle='--')  # Zeichnet das untere Bollinger-Band als gestrichelte, lila Linie.  
    if 'ML_Prediction' in data.columns:  # Prüft, ob ML-Vorhersagen in den Daten vorhanden sind.  
        ax.plot(data.index, data['ML_Prediction'] * data['Close'].shift(1) + data['Close'].shift(1),  
                label='ML Vorhersage (Preis)', color='magenta', linestyle=':')  # Zeichnet die ML-Vorhersage als magentafarbene, gepunktete Linie.  
    ax.set_title(title)  # Setzt den Titel des Diagramms.  
    ax.legend()  # Fügt eine Legende zum Diagramm hinzu.  
    canvas = FigureCanvasTkAgg(fig, master=root)  # Bindet die Matplotlib-Figur in die tkinter-GUI ein.  
    canvas.draw()  # Zeichnet die Figur in der GUI.  
    canvas.get_tk_widget().pack(pady=10)  # Platziert das Diagramm in der GUI mit einem vertikalen Abstand.  

def display_investment_opportunities(data):  # Definiert die Funktion zur Anzeige von Investitionsmöglichkeiten basierend auf technischen Signalen.  
    last_signal = data['Signal'].iloc[-1]  # Liest das letzte Signal aus der 'Signal'-Spalte.  
    status = "Optimal buying opportunity!" if last_signal == 1 else "Currently no buying opportunity."  # Setzt den Status je nach Signal (1 = Kauf, 0 = kein Kauf).  
    messagebox.showinfo("Investment Opportunities (technical)", status)  # Zeigt eine Informationsbox mit dem Investitionsstatus an.  

def display_quant_investment_opportunities(data):  # Definiert die Funktion zur Anzeige von quantitativen Investitionsmöglichkeiten.  
    last_signal = data['Quant_Signal'].iloc[-1]  # Liest das letzte quantitative Signal aus der 'Quant_Signal'-Spalte.  
    status = "Quantitative Buy Signal!" if last_signal == 1 else "No Quantitative Buy Signal."  # Setzt den Status basierend auf dem quantitativen Signal.  
    messagebox.showinfo("Investment Opportunities (quant)", status)  # Zeigt eine Informationsbox mit dem quantitativen Investitionsstatus an.  

# =============================================================================  
# GRAFISCHE BENUTZEROBERFLÄCHE (GUI).  
# =============================================================================  

def create_gui():  # Definiert die Funktion zur Erstellung der grafischen Benutzeroberfläche.  
    global root, entry_ticker  # Deklariert root und entry_ticker als globale Variablen, damit sie in anderen Funktionen zugänglich sind.  

    root = tk.Tk()  # Initialisiert das Hauptfenster der tkinter-GUI.  
    root.title("Home: ZenithAlpha - Quantitative Investment Tool for Hedgefunds and Investors")  # Setzt den Fenstertitel.  
    root.configure(bg="#f0f0f0")  # Setzt die Hintergrundfarbe des Fensters.  
    default_font = ("Helvetica", 12)  # Definiert die Standardschriftart und -größe für die GUI.  

    # Dynamische Anpassung der Fenstergröße.  
    screen_width = root.winfo_screenwidth()  # Ermittelt die Bildschirmbreite.  
    screen_height = root.winfo_screenheight()  # Ermittelt die Bildschirmhöhe.  
    window_width = int(screen_width * 0.8)  # Berechnet 80 % der Bildschirmbreite als Fensterbreite.  
    window_height = int(screen_height * 0.8)  # Berechnet 80 % der Bildschirmhöhe als Fensterhöhe.  
    position_top = int(screen_height / 2 - window_height / 2)  # Berechnet die vertikale Position, um das Fenster zu zentrieren.  
    position_right = int(screen_width / 2 - window_width / 2)  # Berechnet die horizontale Position, um das Fenster zu zentrieren.  
    root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')  # Setzt die Fenstergröße und -position.  

    # Willkommensmeldung in der GUI.  
    welcome_label = tk.Label(root, text="Welcome by ZenithAlpha from Linoz Developments. Version 1.1. Developer: Lukas Völzing", font=("Helvetica", 18, "bold"), bg="#f0f0f0", fg="#000000")  # Erstellt ein Label mit einer Willkommensnachricht, festgelegter Schriftart, Hintergrund- und Schriftfarbe.  
    welcome_label.pack(pady=20)  # Platziert das Label im Fenster mit einem vertikalen Abstand.  

    # Frame für Logos.  
    logo_frame = tk.Frame(root, bg="#f0f0f0")  # Erstellt ein Frame im Hauptfenster für die Logos mit der angegebenen Hintergrundfarbe.  
    logo_frame.pack(pady=10)  # Platziert das Logo-Frame mit einem vertikalen Abstand.  

    # Lade und zeige das erste Bild (ZenithAlpha Logo).  
    logo_path_1 = "D:\\Weiteres\\Privat\\Homepageordner\\HTML und JAVA-Testseiten und Projekte\\X) Private Projekte\\2) Python\\1) Zenith Alpha\\Z) Logo.png"  # Definiert den Pfad zum ersten Logo-Bild.  
    if os.path.exists(logo_path_1):  # Prüft, ob der erste Logo-Pfad existiert.  
        try:  # Beginnt einen Try-Block, um Fehler beim Laden des Bildes abzufangen.  
            logo_image_1 = Image.open(logo_path_1)  # Öffnet das erste Logo-Bild.  
            logo_image_1 = logo_image_1.resize((330, 330), Image.Resampling.LANCZOS)  # Ändert die Größe des Bildes auf 330x330 Pixel unter Verwendung eines hochwertigen Resampling-Verfahrens.  
            logo_photo_1 = ImageTk.PhotoImage(logo_image_1)  # Wandelt das Bild in ein Format um, das von tkinter verwendet werden kann.  
            logo_label_1 = tk.Label(logo_frame, image=logo_photo_1, bg="#f0f0f0")  # Erstellt ein Label im logo_frame, um das erste Logo anzuzeigen, und setzt die Hintergrundfarbe.  
            logo_label_1.image = logo_photo_1  # Verhindert, dass das Bild vom Garbage Collector gelöscht wird, indem es an das Label gebunden wird.  
            logo_label_1.pack(side=tk.LEFT, padx=10)  # Platziert das Label im Frame auf der linken Seite mit horizontalem Abstand.  
        except Exception as e:  # Fängt Fehler ab, die beim Laden des Bildes auftreten könnten.  
            print(f"Error loading the first logo: {e}")  # Gibt eine Fehlermeldung aus, wenn das erste Logo nicht geladen werden kann.  
    else:  # Falls der Pfad zum ersten Logo nicht existiert:  
        print("First logo not found, skipping.")  # Gibt eine Meldung aus, dass das erste Logo übersprungen wird.  

    # Lade und zeige das zweite Bild (Linoz Developments Logo).  
    logo_path_2 = "D:\\Weiteres\\Privat\\Homepageordner\\HTML und JAVA-Testseiten und Projekte\\X) Private Projekte\\2) Python\\1) Zenith Alpha\\Z) Linoz Developments.png"  # Definiert den Pfad zum zweiten Logo-Bild.  
    if os.path.exists(logo_path_2):  # Prüft, ob der Pfad zum zweiten Logo existiert.  
        try:  # Beginnt einen Try-Block, um Fehler beim Laden des Bildes abzufangen.  
            logo_image_2 = Image.open(logo_path_2)  # Öffnet das zweite Logo-Bild.  
            logo_image_2 = logo_image_2.resize((330, 260), Image.Resampling.LANCZOS)  # Ändert die Größe des Bildes auf 330x260 Pixel unter Verwendung eines hochwertigen Resampling-Verfahrens.  
            logo_photo_2 = ImageTk.PhotoImage(logo_image_2)  # Wandelt das Bild in ein tkinter-kompatibles Format um.  
            logo_label_2 = tk.Label(logo_frame, image=logo_photo_2, bg="#f0f0f0")  # Erstellt ein Label im logo_frame, um das zweite Logo anzuzeigen, und setzt die Hintergrundfarbe.  
            logo_label_2.image = logo_photo_2  # Bindet das Bild an das Label, um das Löschen durch den Garbage Collector zu verhindern.  
            logo_label_2.pack(side=tk.LEFT, padx=10)  # Platziert das Label im Frame auf der linken Seite mit horizontalem Abstand.  
        except Exception as e:  # Fängt Fehler ab, die beim Laden des Bildes auftreten könnten.  
            print(f"Error loading the second logo: {e}")  # Gibt eine Fehlermeldung aus, wenn das zweite Logo nicht geladen werden kann.  
    else:  # Falls der Pfad zum zweiten Logo nicht existiert:  
        print("Second logo not found, skipping.")  # Gibt eine Meldung aus, dass das zweite Logo übersprungen wird.  

    # Frame für Eingabefeld und Schaltflächen.  
    input_frame = tk.Frame(root, bg="#f0f0f0")  # Erstellt ein Frame im Hauptfenster für das Eingabefeld und die Schaltflächen mit der angegebenen Hintergrundfarbe.  
    input_frame.pack(pady=20)  # Platziert das Eingabe-Frame mit einem vertikalen Abstand.  

    # Erstelle ein Label und ein Eingabefeld für den Aktien-Ticker.  
    label_ticker = tk.Label(input_frame, text="Enter stock ticker:", font=default_font, bg="#f0f0f0")  # Erstellt ein Label im input_frame mit der Aufforderung, einen Aktien-Ticker einzugeben, und setzt Schriftart und Hintergrundfarbe.  
    label_ticker.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)  # Platziert das Label in der ersten Zeile und Spalte des Grids, mit etwas Abstand und linksbündig.  
    entry_ticker = tk.Entry(input_frame, font=default_font, width=30)  # Erstellt ein Eingabefeld im input_frame mit der angegebenen Schriftart und Breite.  
    entry_ticker.grid(row=0, column=1, padx=5, pady=5)  # Platziert das Eingabefeld in der ersten Zeile, zweite Spalte des Grids, mit etwas Abstand.  

    # Anpassung der Schaltflächen für ein besseres Aussehen.  
    btn_analyze = tk.Button(input_frame, text="Technical Analysis & Backtest", command=analyze_and_plot, font=default_font, bg="#4CAF50", fg="white")  # Erstellt einen Button für die technische Analyse und Backtesting, mit festgelegtem Text, Befehl, Schriftart, Hintergrund- und Schriftfarbe.  
    btn_analyze.grid(row=1, column=0, padx=5, pady=10, sticky=tk.E+tk.W)  # Platziert den Button in der zweiten Zeile, ersten Spalte des Grids, mit Abstand und streckt ihn horizontal.  
    btn_fundamental = tk.Button(input_frame, text="Fundamental Analysis", command=analyze_fundamentals, font=default_font, bg="#2196F3", fg="white")  # Erstellt einen Button für die Fundamentalanalyse mit den entsprechenden Parametern.  
    btn_fundamental.grid(row=1, column=1, padx=5, pady=10, sticky=tk.E+tk.W)  # Platziert den Button in der zweiten Zeile, zweiten Spalte des Grids, mit Abstand und streckt ihn horizontal.  
    btn_quant = tk.Button(input_frame, text="Quantitative Strategy (ML)", command=analyze_quant_strategy, font=default_font, bg="#FF9800", fg="white")  # Erstellt einen Button für die quantitative Strategie (ML) mit den entsprechenden Parametern.  
    btn_quant.grid(row=1, column=2, padx=5, pady=10, sticky=tk.E+tk.W)  # Platziert den Button in der zweiten Zeile, dritten Spalte des Grids, mit Abstand und streckt ihn horizontal.  

    root.mainloop()  # Startet die tkinter-Ereignisschleife, wodurch die GUI interaktiv wird.  

# =============================================================================  
# HAUPTPROGRAMM STARTEN.  
# =============================================================================  

if __name__ == "__main__":  # Prüft, ob dieses Skript als Hauptprogramm ausgeführt wird.  
    create_gui()  # Ruft die Funktion zur Erstellung der GUI auf und startet das Programm.