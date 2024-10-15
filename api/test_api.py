# Tests unitaires de l'API
# test_api.py
import pytest
import requests

def test_api_predict():
    url = 'http://localhost:5000/predict'
    
    # Ajuster les données envoyées à l'API selon vos features
    data = {
        'AGE': 30,
        'INCOME': 53107,
        'CREDIT_SCORE': 381,
        'LOAN_AMOUNT': 15694,
        'DURATION': 45
    }

    # Envoyer la requête POST à l'API
    response = requests.post(url, json=data)
    
    # Vérifier que la réponse est réussie (status code 200)
    assert response.status_code == 200
    
    # Obtenir la réponse au format JSON
    result = response.json()

    # Vérifier que la réponse contient les clés 'prediction' et 'probability'
    assert 'prediction' in result
    assert 'probability' in result
    
    # Vérifier que la prédiction et la probabilité ont des valeurs valides
    assert isinstance(result['prediction'], int)
    assert isinstance(result['probability'], list)
    assert len(result['probability']) > 0


