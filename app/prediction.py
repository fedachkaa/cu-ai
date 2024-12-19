from app.model import load_model

def make_prediction(data):
    model = load_model()
    features = [list(data.values())]
    return model.predict(features).tolist()