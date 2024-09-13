from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Carregar o pipeline treinado
pipeline = joblib.load('modelo_final.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obter os dados enviados no formato JSON
        data = request.get_json(force=True)

        # Definir todas as colunas esperadas, incluindo as codificadas
        expected_columns = [
            'Número de Leitos', 'Avaliação de Qualidade', 'Taxas de Mortalidade',
            'Taxas de Readmissão', 'Tempo de Espera', 'Recursos e Equipamentos',
            'Número de Médicos', 'Número de Enfermeiros', 'Distância até o Centro da Cidade',
            'Taxa de Ocupação dos Leitos', 'Índice de Recursos Médicos', 'Diversidade de Especialidades Médicas',
            'Densidade Populacional', 'Especialidades Médicas_Cardiologia', 'Especialidades Médicas_Neurologia',
            'Especialidades Médicas_Oncologia', 'Especialidades Médicas_Ortopedia', 'Especialidades Médicas_Pediatria'
        ]

        # Converter os dados recebidos em DataFrame
        df = pd.DataFrame([data])

        # Verificar e preencher as colunas ausentes
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        # Reordenar colunas para garantir que estejam na ordem esperada pelo modelo
        df = df[expected_columns]

        # Fazer a previsão com o pipeline
        prediction = pipeline.predict(df)

        # Retornar a previsão
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        # Retornar erro em caso de exceção
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is running'}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
