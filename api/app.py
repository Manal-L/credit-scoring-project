from flask import Flask, render_template, request, jsonify
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)

# Charger le modèle sauvegardé
model_path = "C:/Users/HP/Documents/credit-scoring-project/models/mlruns/208211513410033270/4723a51ef4094da09ff22fd85d00b0b1/artifacts/model"
model = mlflow.pyfunc.load_model(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Obtenir les données du formulaire HTML
        age = request.form['AGE']
        income = request.form['INCOME']
        credit_score = request.form['CREDIT_SCORE']
        loan_amount = request.form['LOAN_AMOUNT']
        duration = request.form['DURATION']

        # Préparer les données dans un DataFrame
        data = {
            'AGE': [int(age)],
            'INCOME': [int(income)],
            'CREDIT_SCORE': [int(credit_score)],
            'LOAN_AMOUNT': [int(loan_amount)],
            'DURATION': [int(duration)]
        }
        df = pd.DataFrame(data)

        # Faire une prédiction
        try:
            prediction = model.predict(df)[0]  # Prendre la première prédiction
            result_message = f"La prédiction du modèle est : {prediction}"
        except Exception as e:
            result_message = f"Erreur lors de la prédiction : {str(e)}"

        # Renvoyer le résultat au template HTML
        return render_template('result.html', prediction=result_message)

    # Si méthode GET, afficher le formulaire
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

