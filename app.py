from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Carregar o pipeline treinado
pipeline = joblib.load('modelo_final.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        expected_columns = [
            'Número de Leitos', 'Avaliação de Qualidade', 'Taxas de Mortalidade',
            'Taxas de Readmissão', 'Tempo de Espera', 'Recursos e Equipamentos',
            'Número de Médicos', 'Número de Enfermeiros', 'Distância até o Centro da Cidade',
            'Taxa de Ocupação dos Leitos', 'Índice de Recursos Médicos', 'Diversidade de Especialidades Médicas',
            'Especialidades Médicas_Cardiologia', 'Especialidades Médicas_Neurologia',
            'Especialidades Médicas_Oncologia', 'Especialidades Médicas_Ortopedia', 'Especialidades Médicas_Pediatria'
        ]

        df = pd.DataFrame([data])

        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_columns]
        prediction = pipeline.predict(df)

        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is running'}), 200


# Rota para exibir os endpoints e exemplos de requisição
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
