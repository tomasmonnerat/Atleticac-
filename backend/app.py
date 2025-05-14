# api/index.py
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API de Previsão de Vendas online."})

@app.route("/prever", methods=["POST"])
def prever():
    try:
        # Receber dados JSON
        data = request.get_json()
        dias = int(data.get("dias", 7))

        # Simula leitura de dados
        df = pd.read_csv("dados.csv")  # você já tem esse arquivo
        df['Dia'] = np.arange(len(df))

        X = df[['Dia']]
        y = df['Vendas']
        modelo = LinearRegression()
        modelo.fit(X, y)

        ultimos_dias = np.arange(len(df), len(df) + dias).reshape(-1, 1)
        previsoes = modelo.predict(ultimos_dias).tolist()

        return jsonify({"previsao": previsoes})
    
    except Exception as e:
        return jsonify({"erro": str(e)}), 500
